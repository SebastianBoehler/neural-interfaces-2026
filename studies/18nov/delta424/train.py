"""Train A6 against the explicit teacher's response to a replaced source prefix."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

BUNDLE = Path.home() / 'ni26/382/bundle'
sys.path.insert(0, str(BUNDLE))
from kernel import configure, group_digest, schedule, sha_array

BUNDLE_SHA = '06f45ef836fab01845cbb007bb5567dc1c39b4835b1b84b4930fdac1f4d6545f'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def episodes(streams, seed, steps):
    owners = np.array([s.split('|')[0] for s in streams])
    rng = np.random.default_rng([424, seed])
    for real in schedule(streams, seed, steps):
        donor = rng.choice(np.unique(streams[owners != owners[real[0]]]))
        prefix = rng.choice(np.flatnonzero(streams == donor), 24, replace=False)
        other = np.r_[prefix, real[24:]]
        assert np.all(owners[prefix] != owners[real[0]]) and len(set(real)) == 48
        yield real, other


def delta(real, other):
    diff = real[24:] - other[24:]
    return diff - diff.mean(dim=1, keepdim=True)


def main(arm, seed, smoke):
    assert arm in ('T','S','C') and seed in (382,1382)
    assert sha(BUNDLE/'bundle.json') == BUNDLE_SHA and 'H100' in torch.cuda.get_device_name()
    meta = json.loads((BUNDLE/'bundle.json').read_text())
    assert meta['source_only'] and meta['people'] == ['P04','P05','P06','P07','P08','P10']
    for k,v in meta['files'].items():
        assert sha(BUNDLE/k) == v
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    net, teacher = [torch.jit.load(str(BUNDLE/(n+'.ts')),map_location='cuda').eval()
                    for n in ('learned','native')]
    groups = configure(net,'J')
    params = [p for ps in groups.values() for p in ps]
    frozen = [p for p in net.parameters() if not p.requires_grad]
    fixed = group_digest(frozen)
    before = {g:group_digest(ps) for g,ps in groups.items()}
    opt = torch.optim.AdamW([dict(params=ps,lr=.0003 if g=='norm' else .00003)
                             for g,ps in groups.items()],weight_decay=.001)
    with np.load(BUNDLE/'source.npz') as z:
        streams = z['streams']
        x,y,w = [torch.from_numpy(z[k]).cuda() for k in ('x','y','weights')]
    with torch.no_grad(), np.load(BUNDLE/'probes.npz') as z:
        b = schedule(streams,382,1)[0]
        for model,key in [(net,'learned'),(teacher,'native')]:
            v = model(x[b])
            ref = torch.from_numpy(z[key]).cuda()
            assert (v-ref).abs().max()<.005 and torch.equal(v.argmax(1),ref.argmax(1))
    pairs = list(episodes(streams,seed,4 if smoke else 120))
    for step,(r,d) in enumerate(pairs):
        opt.zero_grad()
        real = net(x[r])
        ce = (F.cross_entropy(real[24:],y[r[24:]],reduction='none')*w[r[24:]]).mean()
        if arm == 'C':
            ce.backward()
        else:
            with torch.no_grad():
                target = delta(teacher(x[r]),teacher(x[d]))
                shifted = target.roll(12,0)
                assert (target-shifted).square().mean() > 1e-9
            pred = delta(real,net(x[d]))
            loss = F.mse_loss(pred,target if arm=='T' else shifted)
            gc = torch.autograd.grad(ce,params,retain_graph=True)
            ga = torch.autograd.grad(loss,params,retain_graph=step==0)
            if step == 0:
                alt = torch.autograd.grad(F.mse_loss(pred,shifted if arm=='T' else target),params)
                assert sum((a-b).square().sum() for a,b in zip(ga,alt))>1e-10
            nc,na = [sum(g.square().sum() for g in gs).sqrt() for gs in (gc,ga)]
            assert min(nc,na)>1e-10
            for p,c,a in zip(params,gc,ga):
                p.grad = c+.35*(nc/na)*a
        assert all(torch.isfinite(p.grad).all() for p in params)
        if step == 0:
            gradient = {g:group_digest([p.grad for p in ps]) for g,ps in groups.items()}
        opt.step()
    after = {g:group_digest(ps) for g,ps in groups.items()}
    assert all(before[g]!=after[g] for g in groups) and group_digest(frozen)==fixed
    errs = []
    with torch.no_grad():
        for r,d in episodes(streams,4240,24):
            errs.append(float(F.mse_loss(delta(net(x[r]),net(x[d])),delta(teacher(x[r]),teacher(x[d])))))
    key = ('smoke_' if smoke else '')+f'{arm}_{seed}'
    rec = dict(arm=arm,seed=seed,rows=meta['source_rows_sha256'],bundle=BUNDLE_SHA,
               device=torch.cuda.get_device_name(),steps=len(pairs),episodes=sha_array(pairs),
               gradient=gradient,before=before,after=after,fixed=fixed,source=sha(__file__),
               probe_mse=float(np.mean(errs)))
    if not smoke:
        path = Path(key+'.pt')
        torch.save({k:v.detach().cpu() for k,v in net.state_dict().items()},path)
        rec['weights'] = sha(path)
    Path(key+'.json').write_text(json.dumps(rec,indent=2))


if __name__ == '__main__':
    main(sys.argv[1],int(sys.argv[2]),sys.argv[3:]==['smoke'])
