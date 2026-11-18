"""Match teacher-logit and parameter anchor gradient strength on source episodes."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F
BUNDLE = Path.home()/'ni26/382/bundle'
sys.path.insert(0,str(BUNDLE))
from kernel import configure,group_digest,schedule,sha_array

BUNDLE_SHA = '06f45ef836fab01845cbb007bb5567dc1c39b4835b1b84b4930fdac1f4d6545f'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(arm,seed,smoke):
    body,mode = arm
    assert body in 'JK' and mode in 'TPU' and seed in (382,1382) and 'H100' in torch.cuda.get_device_name()
    assert sha(BUNDLE/'bundle.json')==BUNDLE_SHA
    meta=json.loads((BUNDLE/'bundle.json').read_text())
    for k,v in meta['files'].items():
        assert sha(BUNDLE/k)==v
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    name = 'learned' if body=='J' else 'native'
    student = torch.jit.load(str(BUNDLE/(name+'.ts')),map_location='cuda').eval()
    teacher = torch.jit.load(str(BUNDLE/'native.ts'),map_location='cuda').eval()
    groups=configure(student,body)
    params=[p for ps in groups.values() for p in ps]
    initial=[p.detach().clone() for p in params]
    frozen=[p for p in student.parameters() if not p.requires_grad]
    fixed=group_digest(frozen)
    before={g:group_digest(ps) for g,ps in groups.items()}
    opt=torch.optim.AdamW([dict(params=ps,lr=.0003 if g=='norm' else .00003) for g,ps in groups.items()],weight_decay=.001)
    with np.load(BUNDLE/'source.npz') as z:
        streams=z['streams']
        x,y,w=[torch.from_numpy(z[k]).cuda() for k in ('x','y','weights')]
    batches=schedule(streams,seed,120)
    with torch.no_grad(),np.load(BUNDLE/'probes.npz') as z:
        b=schedule(streams,382,1)[0]
        for net,label in [(student,name),(teacher,'native')]:
            real=net(x[b])
            ref=torch.from_numpy(z[label]).cuda()
            assert (real-ref).abs().max()<.005 and torch.equal(real.argmax(1),ref.argmax(1))
    ratios=[]
    first={}
    for i,b in enumerate(batches[:4 if smoke else 120]):
        opt.zero_grad()
        logits=student(x[b])
        ce=(F.cross_entropy(logits,y[b],reduction='none')*w[b]).mean()
        if i and mode!='U':
            gc=torch.autograd.grad(ce,params,retain_graph=True)
            if mode=='T':
                with torch.no_grad():
                    target=(teacher(x[b])/2).softmax(1)
                anchor=4*F.kl_div((logits/2).log_softmax(1),target,reduction='batchmean')
            else:
                anchor=sum((p-ref).square().sum() for p,ref in zip(params,initial))
            ga=torch.autograd.grad(anchor,params)
            nc=sum(g.square().sum() for g in gc).sqrt()
            na=sum(g.square().sum() for g in ga).sqrt()
            assert torch.isfinite(nc+na) and na>1e-12 and nc>1e-12
            scale=.35*nc/na
            for p,a,c in zip(params,ga,gc):
                p.grad=c+scale*a
            ratios.append(float(scale*na/nc))
        else:
            ce.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        if i==0:
            first={g:max(float(p.grad.abs().max()) for p in ps) for g,ps in groups.items()}
        opt.step()
    after={g:group_digest(ps) for g,ps in groups.items()}
    assert all(first[g]>0 and after[g]!=before[g] for g in groups) and group_digest(frozen)==fixed
    assert mode=='U' or np.allclose(ratios,.35,atol=1e-6)
    kl,loss,agree=[],[],[]
    with torch.no_grad():
        for b in schedule(streams,3940,24):
            logits=student(x[b])
            target=teacher(x[b])
            kl.append(float(F.kl_div((logits/2).log_softmax(1),(target/2).softmax(1),reduction='batchmean')*4))
            loss.append(float((F.cross_entropy(logits,y[b],reduction='none')*w[b]).mean()))
            agree.append(float((logits.argmax(1)==target.argmax(1)).float().mean()))
    key=('smoke_' if smoke else '')+arm+'_'+str(seed)
    rec=dict(arm=arm,seed=seed,source=sha(Path('source.sha256')),bundle=BUNDLE_SHA,
        rows=meta['source_rows_sha256'],people=meta['people'],held_labels_opened=False,device=torch.cuda.get_device_name(),
        episodes=sha_array(batches),first=first,before=before,after=after,fixed=fixed,
        steps=4 if smoke else 120,ratio_mean=float(np.mean(ratios)) if ratios else None,active=len(ratios),
        source_probe=dict(kl=float(np.mean(kl)),ce=float(np.mean(loss)),agreement=float(np.mean(agree))))
    if not smoke:
        path=Path(key+'.pt')
        torch.save({k:p.detach().cpu() for k,p in student.state_dict().items()},path)
        rec['weights']=sha(path)
    Path(key+'.json').write_text(json.dumps(rec,indent=2))


if __name__=='__main__':
    main(sys.argv[1],int(sys.argv[2]),sys.argv[3:]==['smoke'])
