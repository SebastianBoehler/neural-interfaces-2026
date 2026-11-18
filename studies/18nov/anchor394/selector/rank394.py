"""Use independent held waveform judgments to rank rows within fixed CP confidence bands."""
import json
import sys
import os
from pathlib import Path
import numpy as np
ROOT=Path('/mnt/lustre/work/gehler/gfh527')
BAND=ROOT/'ni26_band390_v1_20261118'
SOURCE=ROOT/'ni26_anchor394_v1_20261118'
sys.path.insert(0,str(BAND))
from worker import INPUT,SCHEDULE,check,digest,setup
from selector import exchange

PEOPLE=('P01','P02','P03','P09')
SEEDS=(382,1382)
ARMS=('Z','N','JT','JP','JU','KT','KP','KU','D')


def run(model,cfg,feat,raw,filt,wave,arm,seed,stop=None):
    edits=[]
    class CP(model):
        def select(self,y,c):
            stock=super().select(y,c)
            if arm=='Z':
                active=stock
            elif arm=='N':
                active=exchange(y,c,stock,'N'+str(seed+4),None)
            else:
                chosen=np.zeros(len(y),dtype=bool)
                chosen[stock]=True
                p=wave[np.arange(len(y)),y].copy()
                for label in np.unique(y):
                    r=np.flatnonzero(y==label)
                    keep,out=r[chosen[r]],r[~chosen[r]]
                    m=min(len(out),int(len(keep)/8+.5))
                    if not m:
                        continue
                    if arm=='D':
                        rng=np.random.default_rng([394,int(label),len(y)])
                        p[r]=p[rng.permutation(r)]
                    drop=keep[np.argsort(c[keep],kind='stable')[:2*m]]
                    add=out[np.argsort(-c[out],kind='stable')[:2*m]]
                    chosen[drop[np.argsort(p[drop],kind='stable')[:m]]]=False
                    chosen[add[np.argsort(-p[add],kind='stable')[:m]]]=True
                active=np.flatnonzero(chosen)
            check(np.array_equal(np.bincount(y[stock]),np.bincount(y[active])),'budget')
            edits.append(len(np.setdiff1d(active,stock)))
            return active
        def __del__(self):
            pass
    cp=CP(cfg)
    cp.feats=feat.__getitem__
    pred=np.stack([cp.step(i,r,f,i==stop) for i,(r,f) in enumerate(zip(raw,filt,strict=True))])
    h=len(raw) if stop is None else stop
    check(not cp.dead and not cp.log['err'] and cp.refits==len(edits)==len(range(20,h,20)) and cp.fused==max(0,h-90),'activation')
    return pred,edits


def main(smoke,i):
    seed,person=SEEDS[i//4],PEOPLE[i%4]
    prior=json.loads((SOURCE/'result.json').read_text())
    check(digest(SOURCE/'result.json')=='34fddfe515fb793d2df6285285d3e913fa9d9d0835632d940f3f22bb83b8c696'
          and digest(SOURCE/'before_labels.json')==prior['proof'],'member origin')
    ref=json.loads((SOURCE/'before_labels.json').read_text())[person]
    check(digest(SOURCE/(person+'_prob.npz'))==ref['probs'],'member probabilities')
    check(digest(BAND/'source.sha256')=='c6f9f3a964c928a68fb6506081a2e4d0d5541466ee41d5e981d50fab84a5a35f','band source')
    for line in (BAND/'source.sha256').read_text().splitlines():
        h,f=line.split(maxsplit=1)
        check(digest(BAND/f)==h,'band owner')
    assets,cp,cfg=setup()
    feat=np.load(SCHEDULE/'features.npy',mmap_mode='r')
    with np.load(assets.CHAIN/'data/proteus_windows.npz') as z:
        people=np.array([p.rsplit('/',1)[-1] for p in z['subject']])
    target=np.flatnonzero(people==person)
    pack={}
    seal=json.loads((BAND/'before_labels.json').read_text())
    with np.load(INPUT/'inputs.npz') as ctl,np.load(INPUT/'outputs.npz') as native,np.load(SOURCE/(person+'_prob.npz')) as bank:
        check(np.array_equal(target,bank['index']),'row owner')
        for oi,o in enumerate(ctl['orders'][:1] if smoke else ctl['orders']):
            name=f'core_{2*sorted(set(people)).index(person)+oi%2}'
            check(digest(BAND/(name+'.npz'))==seal['new'][name],'band held output')
            parts=np.split(ctl[o+'|index'],ctl[o+'|bounds'][1:-1])
            matches=[(sid,rr) for sid,rr in enumerate(parts) if np.array_equal(np.sort(rr),target)]
            check(len(matches)==1,'person/session mismatch')
            sid,rr=matches[0]
            pos=np.searchsorted(target,rr)
            raw,filt,gold=[ctl[o+'|'+k][rr] for k in ('raw','filtered','expected')]
            check(np.array_equal(gold.argmax(1),native[o+'|stock'][rr].argmax(1)),'native zero')
            counts={}
            with np.load(BAND/(name+'.npz')) as old:
                pre=f'{o}|{sid}|'
                check(np.array_equal(old[pre+'rows'],rr),'pinned order')
                for a in ARMS:
                    wave=bank[f'{seed}|'+('JT' if a in ('Z','D','N') else a)+'|'+o][pos]
                    p,t=run(cp,cfg,feat[rr],raw,filt,wave,a,seed)
                    counts[a]=t
                    check(np.array_equal(p[:90],gold[:90]),'premature changes')
                    if a in ('N','Z'):
                        check(np.array_equal(p.argmax(1),old[pre+('Z' if a=='Z' else 'N'+str(seed+4))]),'exact near/stock')
                    if a=='Z':
                        check(np.array_equal(p,gold),'exact stock scores')
                    if smoke and a in ('JT','D'):
                        short,_=run(cp,cfg,feat[rr[:131]],raw[:131],filt[:131],wave[:131],a,seed)
                        stop,_=run(cp,cfg,feat[rr[:131]],raw[:131],filt[:131],wave[:131],a,seed,63)
                        check(np.array_equal(short,p[:131]) and np.array_equal(stop,filt[:131]),'prefix/stop')
                    pack[o+'|'+a]=p.argmax(1)[np.argsort(pos)].astype('int8')
            check(all(t==counts['N'] for a,t in counts.items() if a!='Z'),'turnover differed')
    name='smoke' if smoke else f'{seed}_{person}'
    np.savez_compressed(name+'.npz',index=target,**pack)
    Path(name+'.json').write_text(json.dumps(dict(source=digest('source.sha256'),outputs=digest(name+'.npz'))))


if __name__=='__main__':
    main(sys.argv[1]=='smoke',int(os.environ.get('SLURM_ARRAY_TASK_ID',0)))
