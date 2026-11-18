"""Lock new-seed outputs and score the exact predeclared routing policies."""
import json
from pathlib import Path
import numpy as np
from worker import CORE,INPUT,check,digest,setup
from score386 import summarize


def main():
    assets,_,_=setup()
    source=digest('source.sha256')
    proof={}
    data={}
    for i in range(20):
        n=f'core_{i}'
        card=json.loads(Path(n+'.json').read_text())
        check(card==dict(source=source,outputs=digest(n+'.npz')),'shard seal')
        proof[n]=card['outputs']
        with np.load(n+'.npz') as z:
            check(not data.keys()&set(z.files),'duplicate')
            data.update({k:z[k] for k in z.files})
    check(len(data)==8*14*(1+2*len(CORE)) and Path('replay.log').read_text().splitlines()[-1]=='OK','coverage/replay')
    out={}
    with np.load(INPUT/'inputs.npz') as ctl:
        for o in ctl['orders']:
            filt=ctl[o+'|filtered']
            for s,rr in enumerate(np.split(ctl[o+'|index'],ctl[o+'|bounds'][1:-1])):
                pre=f'{o}|{s}|'
                check(np.array_equal(data[pre+'rows'],rr),'rows')
                stock=data[pre+'Z']
                for k in ('N','F'):
                    a,b=[data[pre+k+str(seed)] for seed in (2390,3390)]
                    same=(a==b)&(a!=stock)
                    for label,active in [('agree',same),('opposed',same&(filt[rr,a]<filt[rr,stock])),
                                         ('confirmed',same&(filt[rr,a]>filt[rr,stock]))]:
                        pred=out.setdefault(o+'|'+k+label,np.full(2655,-1,dtype='int8'))
                        check(np.all(pred[rr]==-1) and not active[:90].any(),'scope/prefix')
                        pred[rr]=np.where(active,a,stock)
                out.setdefault(o+'|Z',np.full(2655,-1,dtype='int8'))[rr]=stock
    check(all((p>=0).all() for p in out.values()),'missing rows')
    np.savez_compressed('policies.npz',**out)
    Path('before_labels.json').write_text(json.dumps(dict(source=source,workers=proof,policies=digest('policies.npz'))))
    scorer,_,people,_,meta,_=assets.load_controls(dtype=np.float32)
    from run import owner
    _,metric=owner()
    with np.load(assets.CHAIN/'data/proteus_windows.npz') as z:
        y=z['y']
    vals={}
    counts=np.zeros(10)
    events={}
    for i,o in enumerate(scorer.ORDERS):
        stock=out[o+'|Z']
        for key,p in out.items():
            order,k=key.split('|')
            if order!=o:
                continue
            ev=events.setdefault(k,np.zeros(4,dtype=int))
            ev += [np.sum(p!=stock),np.sum((p==y)&(p!=stock)),np.sum((stock==y)&(p!=stock)),
                   np.sum((stock!=y)&(p!=y)&(p!=stock))]
            v=vals.setdefault(k,np.empty((8,10)))
            for j,person in enumerate(scorer.PEOPLE):
                m=people==person
                row=metric(y[m],p[m],meta,m)
                counts[j]=row['n_cells']
                v[i,j]=100*row['balanced_accuracy_cell_mean']
    contrasts=[(k,'Z') for k in vals if k!='Z']+[('Nopposed','Fopposed')]
    for a,b in contrasts:
        vals[a+'-'+b]=vals[a]-vals[b]
    old=np.isin(scorer.PEOPLE,['P01','P02','P03','P09'])
    groups={g:{k:summarize(v[:,m],counts[m]) for k,v in vals.items()} for g,m in
            [('all',np.ones(10,dtype=bool)),('other',~old),('exposed',old)]}
    check(abs(groups['all']['Z']['official']-76.27933502145758)<1e-8,'zero')
    Path('result.json').write_text(json.dumps(dict(groups=groups,people=list(scorer.PEOPLE),
        events={k:v.tolist() for k,v in events.items()},proof=digest('before_labels.json')),indent=2))


if __name__=='__main__':
    main()
