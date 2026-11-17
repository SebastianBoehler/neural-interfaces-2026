"""Diagnostic only: predict unseen within-session variance without any labels."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

OWNER=Path('/mnt/lustre/work/gehler/gfh527/ni26_shrink378_v3_20261118')
sys.path.insert(0,str(OWNER))
from produce import PARENT, prepare, SessionBN, components, src, digest

AGES=(1,8,32,90)
ARMS=('K','G','P','RV')


def moments(net, layer, x):
    values=[]
    def collect(module,args):
        y=args[0].double().flatten(2)
        values.append(torch.stack((y.mean(-1),y.square().mean(-1)),dim=1).numpy())
    handle=layer.register_forward_pre_hook(collect)
    try:
        with torch.no_grad():
            for batch in x.split(32):
                net(batch)
    finally:
        handle.remove()
    return np.concatenate(values)


def run(person):
    torch.set_num_threads(2)
    assert src.sha(OWNER/'source.sha256')=='418012cd84e6fd6cd402e75e2577ecd0f6e82a106f3c478d1a9ff8365ae5daef'
    for row in (OWNER/'source.sha256').read_text().splitlines():
        expected,name=row.split(maxsplit=1)
        assert src.sha(OWNER/name)==expected
    prod=json.loads((OWNER/person/'proof.json').read_text())
    assert prod['held']==person and prod['labels_opened'] is False
    rows=[]
    for seed in (0,1):
        d,fit,test,_,perms,nets,_,meta=src.load(person,seed,'cpu')
        assert 'y' not in d and digest(fit)==prod['train_rows'] and digest(test)==prod['test_rows']
        assert meta['checkpoints']['P']==prod['records'][seed]['checkpoint']
        net,scale=nets['P']
        x=torch.tensor(d['X']/scale,dtype=torch.float32)
        vs,_=prepare(net,x[fit[:8]])
        model=vs['V']
        layer,=[m for m in model.body.modules() if isinstance(m,SessionBN)]
        w,t=components(model,layer,x[fit],d['stream'][fit])
        k=np.divide(w,t,out=np.full_like(w,np.inf),where=t>0)
        median=np.exp(np.median(np.log(k[np.isfinite(k)])))
        assert np.isclose(median,prod['records'][seed]['variance_ratio_k_median'],rtol=1e-4)
        mu=layer.running_mean.double().numpy()
        var=layer.running_var.double().numpy()
        estimates={'K':k,'G':np.full_like(k,median),'P':np.roll(k,1),'RV':np.full_like(k,20.)}
        m=moments(model,layer,x[test])
        for order,perm in perms.items():
            for stream in np.unique(d['stream'][test]):
                delivered=perm[np.isin(perm,test[d['stream'][test]==stream])]
                item=m[np.searchsorted(test,delivered)]
                for n in AGES:
                    prior,future=item[:n],item[n:n+32]
                    assert len(future)==32 and not np.intersect1d(delivered[:n],delivered[n:n+32]).size
                    m1,m2=future.mean(0)
                    target=m2-m1*m1
                    sample1,sample2=prior.mean(0)
                    assert np.all(target>0)
                    for arm,kappa in estimates.items():
                        a=n/(n+kappa)
                        estimate=(1-a)*(var+mu*mu)+a*sample2-((1-a)*mu+a*sample1)**2
                        assert np.all(estimate>0)
                        error=np.log(estimate/target)
                        rows.append([seed,order,stream,n,arm,np.mean(error*error),np.mean(abs(error))])
    assert len(rows)==2*8*len(np.unique(d['stream'][test]))*len(AGES)*len(ARMS)
    Path(person+'.json').write_text(json.dumps(dict(person=person,rows=rows,held_labels_used=False,
        source_sha=src.sha(Path('source.sha256')),source_proof=src.sha(OWNER/person/'proof.json'))))


if __name__=='__main__':
    run(sys.argv[1])
