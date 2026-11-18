"""Train a causal residual while freezing the complete original A6."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from adapter import Residual,SourceModel
BUNDLE=Path.home()/'ni26/382/bundle'
sys.path.insert(0,str(BUNDLE))
from kernel import schedule,group_digest,sha_array
from fit_bundle import sha


def main(arm,seed,smoke):
    assert arm in ('RT','RP','RU','CT') and seed in (382,1382) and 'H100' in torch.cuda.get_device_name()
    meta=json.loads((BUNDLE/'bundle.json').read_text())
    assert sha(BUNDLE/'bundle.json')=='06f45ef836fab01845cbb007bb5567dc1c39b4835b1b84b4930fdac1f4d6545f'
    for k,h in meta['files'].items():
        assert sha(BUNDLE/k)==h
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    original=torch.jit.load(str(BUNDLE/'native.ts'),map_location='cuda').eval()
    for p in original.parameters():
        p.requires_grad_(False)
    fixed=group_digest(original.parameters())
    torch.manual_seed(398)
    adapter=Residual(arm[0]=='R').cuda()
    net=SourceModel(original,adapter).eval()
    params=list(adapter.parameters())
    initial=[p.detach().clone() for p in params]
    start=group_digest(params)
    optimizer=torch.optim.AdamW(params,lr=.0003,weight_decay=.001)
    with np.load(BUNDLE/'source.npz') as d:
        x,y,w=[torch.from_numpy(d[k]).cuda() for k in ('x','y','weights')]
        streams=d['streams']
    batches=schedule(streams,seed,120)
    b=schedule(streams,382,1)[0]
    with torch.no_grad(),np.load(BUNDLE/'probes.npz') as d:
        gold=original(x[b])
        ref=torch.from_numpy(d['native']).cuda()
        assert (gold-ref).abs().max()<.005 and torch.equal(gold.argmax(1),ref.argmax(1))
        assert torch.equal(net(x[b]),gold)
    ratios=[]
    hidden=[]
    for i,b in enumerate(batches[:5 if smoke else 120]):
        optimizer.zero_grad()
        logits=net(x[b])
        ce=(F.cross_entropy(logits,y[b],reduction='none')*w[b]).mean()
        if i==0 or arm=='RU':
            ce.backward()
        else:
            gc=torch.autograd.grad(ce,params,retain_graph=True)
            if arm[1]=='T':
                with torch.no_grad():
                    target=(original(x[b])/2).softmax(1)
                loss=4*F.kl_div((logits/2).log_softmax(1),target,reduction='batchmean')
            else:
                loss=sum((p-q).square().sum() for p,q in zip(params,initial))
            ga=torch.autograd.grad(loss,params)
            nc=sum(g.square().sum() for g in gc).sqrt()
            na=sum(g.square().sum() for g in ga).sqrt()
            assert torch.isfinite(nc+na) and na>1e-12 and nc>1e-12
            scale=.35*nc/na
            for p,c,a in zip(params,gc,ga):
                p.grad=c+scale*a
            ratios.append(float(scale*na/nc))
        assert all(torch.isfinite(p.grad).all() for p in params)
        hidden.append(float(params[0].grad.abs().max()))
        optimizer.step()
    assert hidden[0]==0 and max(hidden[1:])>0 and fixed==group_digest(original.parameters())
    assert start!=group_digest(params) and (arm=='RU' or np.allclose(ratios,.35,atol=1e-6))
    changes=[]
    ce_probe=[]
    with torch.no_grad():
        for b in schedule(streams,3980,24):
            a,z=net(x[b]),original(x[b])
            changes.append(float((a-z).abs().max()))
            ce_probe.append(float((F.cross_entropy(a,y[b],reduction='none')*w[b]).mean()))
    assert np.isfinite(changes).all() and max(changes)>1e-4
    name=('smoke_' if smoke else '')+f'{arm}_{seed}'
    card=dict(arm=arm,seed=seed,source=sha(Path('source.sha256')),bundle=sha(BUNDLE/'bundle.json'),
        rows=meta['source_rows_sha256'],steps=len(hidden),max_logit_shift=max(changes),source_ce=float(np.mean(ce_probe)),
        episodes=sha_array(batches),fixed=fixed,initial=start,final=group_digest(params),active=len(ratios),
        held_labels_opened=False,device=torch.cuda.get_device_name())
    if not smoke:
        p=Path(name+'.pt')
        torch.save({k:v.detach().cpu() for k,v in adapter.state_dict().items()},p)
        card['weights']=sha(p)
    Path(name+'.json').write_text(json.dumps(card,indent=2))


if __name__=='__main__':
    main(sys.argv[1],int(sys.argv[2]),sys.argv[3:]==['smoke'])
