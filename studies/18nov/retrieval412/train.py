"""Fit a per-window similarity metric on the original A6 source bundle."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from model import Metric, features, pair_loss

bundle = Path.home() / 'ni26/382/bundle'
sys.path.insert(0, str(bundle))
from kernel import group_digest, schedule, sha_array
from fit_bundle import sha


def fit(arm, seed, smoke):
    assert arm in ('real','destroyed') and seed in (412,1412)
    device = torch.cuda.get_device_name()
    assert 'H100' in device
    assert sha(bundle/'bundle.json') == '06f45ef836fab01845cbb007bb5567dc1c39b4835b1b84b4930fdac1f4d6545f'
    meta = json.loads((bundle/'bundle.json').read_text())
    for name, digest in meta['files'].items():
        assert sha(bundle/name) == digest
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    net = torch.jit.load(str(bundle/'native.ts'), map_location='cuda').eval()
    with np.load(bundle/'source.npz') as z:
        x, y = [torch.from_numpy(z[k]).cuda() for k in ('x','y')]
        streams = z['streams']
        assert sha_array(z['source_rows']) == meta['source_rows_sha256']
    batches = schedule(streams, seed, 120)
    with torch.no_grad(), np.load(bundle/'probes.npz') as probe:
        pred = net(x[schedule(streams,382,1)[0]])
        np.testing.assert_allclose(pred.cpu().numpy(), probe['native'], atol=.005, rtol=0)
        # The exported spatial layer operates independently per window.
        keys = []
        for start in range(0,len(x),48):
            block = x[start:start+48]
            padded = torch.cat([block, x[:48-len(block)]])
            keys.append(features(net,padded)[:len(block)])
        raw = torch.cat(keys)
        torch.testing.assert_close(features(net,x[batches[0]]),raw[batches[0]],atol=1e-4,rtol=1e-4)
    mean, scale = raw.mean(0),raw.std(0).clamp_min(1e-4)
    standardized = (raw - mean) / scale
    torch.manual_seed(seed)
    model = Metric(raw.shape[1]).cuda()
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    before = group_digest(model.parameters())
    rng = np.random.default_rng(seed+10000)
    changes, losses = [], []
    for ids in batches[:3 if smoke else 120]:
        assert len(set(streams[ids])) == 1
        labels = y[ids]
        perm = rng.permutation(len(ids))
        target = labels if arm == 'real' else labels[perm]
        changes.append(int((labels != target).sum()))
        optimizer.zero_grad()
        loss = pair_loss(model(standardized[ids]),target)
        loss.backward()
        grad = model.project.weight.grad
        assert torch.isfinite(grad).all() and grad.abs().max() > 0
        optimizer.step()
        losses.append(float(loss.detach()))
    final = group_digest(model.parameters())
    assert before != final and (sum(changes)>0) == (arm=='destroyed')
    if not smoke:
        out = Path(f'{arm}_{seed}.pt')
        torch.save(dict(weight=model.project.weight.detach().cpu(),mean=mean.cpu(),scale=scale.cpu()),out)
        out.with_suffix('.json').write_text(json.dumps(dict(arm=arm,seed=seed,rows=meta['source_rows_sha256'],
            source=sha(Path('source.sha256')),device=device,weights=sha(out),before=before,final=final,
            schedule=sha_array(batches),steps=len(losses),permuted_labels=sum(changes),last_ce=np.mean(losses[-8:]))))


if __name__=='__main__':
    fit(sys.argv[1],int(sys.argv[2]),sys.argv[3:]==['smoke'])
