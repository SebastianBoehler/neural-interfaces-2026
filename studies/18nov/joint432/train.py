"""Jointly optimize an A6 body with explicit normalization and a learned causal residual."""
from pathlib import Path
import json
import sys

import numpy as np
import torch
from torch.nn import functional as F

BUNDLE = Path.home()/'ni26/382/bundle'
sys.path[:0] = [str(BUNDLE), str(Path.home()/'ni26/398')]
from adapter import Residual, SourceModel
from fit_bundle import sha
from kernel import configure, group_digest, schedule, sha_array


def main(arm, seed, smoke):
    assert arm in ('R','K') and seed in (382,1382) and 'H100' in torch.cuda.get_device_name()
    assert sha(BUNDLE/'bundle.json')=='06f45ef836fab01845cbb007bb5567dc1c39b4835b1b84b4930fdac1f4d6545f'
    meta=json.loads((BUNDLE/'bundle.json').read_text())
    assert meta['source_only'] and meta['people']==['P04','P05','P06','P07','P08','P10']
    for k,h in meta['files'].items():
        assert sha(BUNDLE/k)==h
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    original=torch.jit.load(str(BUNDLE/'native.ts'),map_location='cuda').eval()
    groups=configure(original,'K')
    frozen=[p for p in original.parameters() if not p.requires_grad]
    fixed=group_digest(frozen)
    torch.manual_seed(398)
    residual=Residual(True).cuda()
    bypass=SourceModel(original,torch.nn.Identity()).eval()
    net=SourceModel(original,residual).eval() if arm=='R' else bypass
    if arm=='R':
        groups['residual']=list(residual.parameters())
    before={g:group_digest(ps) for g,ps in groups.items()}
    opt=torch.optim.AdamW([dict(params=ps,lr=.0003 if g in ('norm','residual') else .00003)
                           for g,ps in groups.items()],weight_decay=.001)
    with np.load(BUNDLE/'source.npz') as z:
        x,y,w=[torch.from_numpy(z[k]).cuda() for k in ('x','y','weights')]
        streams=z['streams']
    probe=schedule(streams,382,1)[0]
    with np.load(BUNDLE/'probes.npz') as z,torch.no_grad():
        for _ in range(3):
            original(x[probe])
            net(x[probe])
        gold=original(x[probe])
        ref=torch.from_numpy(z['native']).cuda()
        assert (gold-ref).abs().max()<.005 and torch.equal(gold.argmax(1),ref.argmax(1))
        conv,norm,_=original.spatial_conv.children()
        h=norm(conv(original.spectral_filtering(x[probe])))
        assert torch.equal(residual(h),h)
        # The whole compiled trace may fuse operations differently from its children.
        stock=net(x[probe])
        graph_error=float((stock-gold).abs().max())
        torch.testing.assert_close(stock,gold,atol=2e-5,rtol=1e-5)
        assert torch.equal(stock.argmax(1),gold.argmax(1))
    batches=schedule(streams,seed,4 if smoke else 120)
    for i,b in enumerate(batches):
        opt.zero_grad()
        logits=net(x[b])
        loss=(F.cross_entropy(logits,y[b],reduction='none')*w[b]).mean()
        loss.backward()
        if i==0:
            gradient={g:max(float(p.grad.abs().max()) for p in ps) for g,ps in groups.items()}
            assert all(v>0 for v in gradient.values())
        assert all(torch.isfinite(p.grad).all() for ps in groups.values() for p in ps)
        opt.step()
    after={g:group_digest(ps) for g,ps in groups.items()}
    assert all(after[g]!=v for g,v in before.items()) and group_digest(frozen)==fixed
    with torch.no_grad():
        effect=float((net(x[probe])-bypass(x[probe])).abs().max())
    assert (effect>1e-5 if arm=='R' else effect==0)
    key=('smoke_' if smoke else '')+f'{arm}_{seed}'
    rec=dict(arm=arm,seed=seed,source=sha(Path('source.sha256')),steps=len(batches),
             init_trace_error=graph_error,rows=meta['source_rows_sha256'],episodes=sha_array(batches),before=before,
             after=after,gradient=gradient,fixed=fixed,source_residual_effect=effect)
    if not smoke:
        dest=Path(key+'.pt')
        torch.save(dict(native=original.state_dict(),residual=residual.state_dict() if arm=='R' else {}),dest)
        rec['weights']=sha(dest)
    Path(key+'.json').write_text(json.dumps(rec,indent=2))


if __name__=='__main__':
    main(sys.argv[1],int(sys.argv[2]),sys.argv[3:]==['smoke'])
