"""Same-query A6 training with a factorial history partition and auxiliary objective."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F
from episodes import episodes

BUNDLE = Path('../382/bundle').resolve()
sys.path.insert(0, str(BUNDLE))
from fit_bundle import sha
from kernel import configure, group_digest, schedule, sha_array

ROWS = '882b5b97ec811d697540f6152be0d02eb49448e9e4c88c3d6c28fcb1433b2f1f'
ARMS = ('skew_aux','skew_ce','random_aux','random_ce')


def objectives(net, x, y, weights, prefixes, query):
    p, q = (net(x[np.r_[prefix,query]])[24:] for prefix in prefixes)
    ce = ((F.cross_entropy(p,y[query],reduction='none') +
           F.cross_entropy(q,y[query],reduction='none')) * weights[query]).mean() / 2
    diff = p - q
    return ce, ((diff - diff.mean(1,keepdim=True)) ** 2).mean()


def norm(grads):
    return torch.stack([g.square().sum() for g in grads]).sum().sqrt()


def run(arm, seed, smoke):
    assert arm in ARMS and 'H100' in torch.cuda.get_device_name()
    meta = json.loads((BUNDLE/'bundle.json').read_text())
    assert meta['source_only'] and meta['source_rows_sha256'] == ROWS
    assert meta['people'] == ['P04','P05','P06','P07','P08','P10']
    for name, digest in meta['files'].items():
        assert sha(BUNDLE/name) == digest
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    with np.load(BUNDLE/'source.npz') as data:
        labels, streams = data['y'], data['streams']
        assert sha_array(data['source_rows']) == ROWS
        x,y,w = [torch.from_numpy(data[k]).cuda() for k in ('x','y','weights')]
    net = torch.jit.load(str(BUNDLE/'learned.ts'),map_location='cuda').eval()
    with np.load(BUNDLE/'probes.npz') as z, torch.no_grad():
        probe = net(x[schedule(streams,382,1)[0]])
        target = torch.from_numpy(z['learned']).cuda()
        assert (probe-target).abs().max() < .005 and torch.equal(probe.argmax(1),target.argmax(1))
    groups = configure(net,'J')
    params = [p for ps in groups.values() for p in ps]
    frozen = [p for p in net.parameters() if not p.requires_grad]
    frozen_sha = group_digest(frozen)
    before = {g:group_digest(ps) for g,ps in groups.items()}
    opt = torch.optim.AdamW([dict(params=ps,lr=.0003 if g=='norm' else .00003)
                             for g,ps in groups.items()],weight_decay=.001)
    digest = hashlib.sha256()
    discrepancy, losses, ratios = [], [], []
    steps = 3 if smoke else 120
    for skew, random, query in episodes(labels,streams,seed,steps):
        for rows in (skew,random,query):
            digest.update(rows.tobytes())
        prefixes = skew if arm.startswith('skew') else random
        counts = [np.bincount(labels[row],minlength=3) for row in prefixes]
        discrepancy.append(np.abs(counts[0]-counts[1]).sum()/48)
        ce, aux = objectives(net,x,y,w,prefixes,query)
        ce_grad = torch.autograd.grad(ce,params,retain_graph=True)
        aux_grad = torch.autograd.grad(aux,params)
        v, u = norm(ce_grad), norm(aux_grad)
        assert torch.isfinite(v+u) and min(v.item(),u.item()) > 0
        ratio = .35 * v/u if arm.endswith('aux') else 0.
        for p,g,h in zip(params,ce_grad,aux_grad):
            p.grad = g + ratio*h
        if not losses:
            first_grad = {g:norm([p.grad for p in ps]).item() for g,ps in groups.items()}
            aux_norm = {g:norm([h for p,h in zip(params,aux_grad) if any(p is v for v in ps)]).item()
                        for g,ps in groups.items()}
            assert all(np.isfinite(v) and v > 0 for v in first_grad.values())
            assert aux_norm['norm'] > 0
        ratios.append(float(ratio*u/v))
        opt.step()
        losses.append([ce.item(),aux.item()])
    assert np.allclose(ratios,.35 if arm.endswith('aux') else 0)
    after = {g:group_digest(ps) for g,ps in groups.items()}
    assert all(after[g]!=before[g] for g in groups) and group_digest(frozen)==frozen_sha
    with torch.no_grad():
        diagnostics = []
        for skew, random, query in episodes(labels,streams,445002,2 if smoke else 24):
            diagnostics.append([[v.item() for v in objectives(net,x,y,w,p,query)] for p in (skew,random)])
    rec = dict(arm=arm,seed=seed,steps=steps,before=before,after=after,first_grad=first_grad,
               aux_norm=aux_norm,applied_aux_ratio=np.mean(ratios),frozen_sha=frozen_sha,
               batches=digest.hexdigest(),rows=ROWS,source=sha(Path('source.sha256')),
               bundle=sha(BUNDLE/'bundle.json'),partition_tv=np.mean(discrepancy),
               first_ce_aux=losses[0],last8_ce_aux=np.mean(losses[-8:],axis=0).tolist(),
               independent_episode_ce_aux=np.mean(diagnostics,axis=0).tolist())
    key = ('smoke_' if smoke else '') + f'{arm}_{seed}'
    if not smoke:
        path = Path(key+'.pt')
        torch.save({k:v.cpu() for k,v in net.state_dict().items()},path)
        rec['checkpoint'] = sha(path)
    Path(key+'.json').write_text(json.dumps(rec,indent=2))


if __name__ == '__main__':
    run(sys.argv[1], int(sys.argv[2]), sys.argv[3:]==['smoke'])
