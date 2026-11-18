"""Count residual output activity at logits, training-set membership and endpoint."""
import json
from pathlib import Path
import numpy as np
from full398 import INPUT,SEEDS,PEOPLE,setup,digest,check


def rank(y,c,stock,p):
    chosen=np.zeros(len(y),dtype=bool)
    chosen[stock]=True
    for label in np.unique(y):
        rows=np.flatnonzero(y==label)
        keep,out=rows[chosen[rows]],rows[~chosen[rows]]
        n=min(len(out),int(len(keep)/8+.5))
        if n:
            drop=keep[np.argsort(c[keep],kind='stable')[:2*n]]
            add=out[np.argsort(-c[out],kind='stable')[:2*n]]
            chosen[drop[np.argsort(p[drop,label],kind='stable')[:n]]]=False
            chosen[add[np.argsort(-p[add,label],kind='stable')[:n]]]=True
    return chosen


def main():
    check(digest('before_labels.json')==json.loads(Path('result.json').read_text())['proof'],'output seals')
    seals=json.loads(Path('before_labels.json').read_text())
    _,cp,cfg=setup()
    stage=cp(cfg)
    out={}
    with np.load(INPUT/'inputs.npz') as ctl:
        for s in SEEDS:
            for arm in ('RT','RP','RU','CT'):
                total=np.zeros(6,dtype=int)
                for p in PEOPLE:
                    n=f'full_{s}_{p}'
                    check(seals[p]['outputs']==digest(p+'.npz') and seals[n]['outputs']==digest(n+'.npz'),'owner')
                    with np.load(p+'.npz') as z,np.load(n+'.npz') as full:
                        target=z['index']
                        for o in ctl['orders']:
                            k=z['A6|'+o]
                            a=z[f'{s}|{arm}|'+o]
                            whole_changed=np.sum(a.argmax(1)!=k.argmax(1))
                            sessions=np.split(ctl[o+'|index'],ctl[o+'|bounds'][1:-1])
                            matched=[rows for rows in sessions if np.array_equal(np.sort(rows),target)]
                            check(len(matched)==1,'session')
                            rr=matched[0]
                            pos=np.searchsorted(target,rr)
                            k,a=k[pos],a[pos]
                            raw=ctl[o+'|raw'][rr]
                            changed,members,refits,events=0,0,0,0
                            for t in range(20,len(rr),20):
                                y=raw[:t].argmax(1)
                                c=raw[:t].max(1)
                                stock=stage.select(y,c)
                                km=rank(y,c,stock,k)
                                am=rank(y,c,stock,a)
                                count=np.sum(am&~km)
                                changed+=count
                                members+=km.sum()
                                events+=count>0
                                refits+=1
                            final=np.sum(full[o+'|'+arm]!=full[o+'|A6'])
                            total += [whole_changed,changed,members,events,refits,final]
                out[f'{s}_{arm}']=dict(zip(('whole_label_changes','substituted_vs_A6','total_kept','different_refits','refits','final_label_changes'),map(int,total)))
    Path('activity.json').write_text(json.dumps(out,indent=2))


if __name__=='__main__':
    main()
