"""Replicate turnover at two fixed doses and average completed CP log scores."""
import json
import os
import sys
from pathlib import Path
import numpy as np

BASE = Path('/mnt/lustre/work/gehler/gfh527/ni26_turn386_v1_20261118')
sys.path.insert(0,str(BASE))
from worker import INPUT,SCHEDULE,check,digest,setup
from score386 import summarize

SEEDS = (2386,3386)
ARMS = ('Z','B','F2386','P2386','F3386','P3386','Fpair','Ppair')


def output(parent,cfg,rows,feat,raw,filt,arm,den,stop=None):
    class Dose(parent):
        def select(self,y,c):
            stock = super().select(y,c)
            if arm=='Z':
                return stock
            chosen = np.zeros(len(y),dtype=bool)
            chosen[stock] = True
            for label in np.unique(y):
                past = np.flatnonzero(y==label)
                kept,out = past[chosen[past]],past[~chosen[past]]
                n = min(len(out),int(len(kept)/den+.5))
                if n:
                    if arm=='B':
                        priority = c
                    else:
                        prefix = len(y) if arm.startswith('F') else 0
                        priority = np.random.default_rng([int(arm[1:]),int(label),prefix]).random(len(y))
                    chosen[kept[np.argsort(priority[kept],kind='stable')[:n]]] = False
                    chosen[out[np.argsort(-priority[out],kind='stable')[:n]]] = True
                check(chosen[past].sum()==len(kept) and chosen[out].sum()==n,'dose budget')
            return np.flatnonzero(chosen)
        def __del__(self):
            pass
    cp = Dose(cfg)
    cp.feats = feat.__getitem__
    p = np.stack([cp.step(r,raw[r],filt[r],i==stop) for i,r in enumerate(rows)])
    horizon = len(rows) if stop is None else stop
    check(not cp.dead and not cp.log['err'] and cp.refits==len(range(20,horizon,20))
          and cp.fused==max(0,horizon-90),'CP activity')
    return p


def main(mode,idx):
    check(digest(BASE/'source.sha256')=='5fe10977cd9e510da3106442a0ffa6ddd94a08f87c137a7cc5654b9197f471b7','parent manifest')
    for line in (BASE/'source.sha256').read_text().splitlines():
        h,n = line.split(maxsplit=1)
        check(digest(BASE/n)==h,'parent changed')
    assets,cp,cfg = setup()
    with np.load(assets.CHAIN/'data/proteus_windows.npz') as z:
        people = np.array([p.rsplit('/',1)[1] for p in z['subject']])
        stream = np.char.add(np.char.add(z['subject'],'/'),z['session'])
    feat = np.load(SCHEDULE/'features.npy',mmap_mode='r')
    by_person = sorted(set(people))
    smoke = mode=='smoke'
    den = (16,4)[idx//20]
    pack = {}
    with np.load(INPUT/'inputs.npz') as z,np.load(INPUT/'outputs.npz') as native,np.load(assets.CHAIN/'runs/shuffle_perms.npz') as perms:
        for order in (z['orders'][:1] if smoke else z['orders'][idx%2::2]):
            raw,filt,expected,indices,bounds = [z[order+'|'+k] for k in ('raw','filtered','expected','index','bounds')]
            perm = perms[f'{order[:-1]}|{order[-1]}|all_loso']
            for sid,rr in enumerate(np.split(indices,bounds[1:-1])):
                if people[rr[0]]!=by_person[idx%20//2]:
                    continue
                key = f'{den}|{order}|{sid}|'
                check(np.array_equal(rr,perm[stream[perm]==stream[rr[0]]]),'pinned order')
                pack[key+'rows'] = rr
                scores = {}
                for arm in ARMS[:6]:
                    scores[arm] = output(cp,cfg,rr,feat,raw,filt,arm,den)
                    check(np.array_equal(scores[arm][:90],expected[rr[:90]]),'premature action')
                    if smoke and arm!='Z':
                        q = output(cp,cfg,rr[:131],feat,raw,filt,arm,den)
                        stop = output(cp,cfg,rr[:131],feat,raw,filt,arm,den,63)
                        check(np.array_equal(q,scores[arm][:131]) and np.array_equal(stop,filt[rr[:131]]),'causal smoke')
                check(np.array_equal(scores['Z'],expected[rr]) and
                      np.array_equal(scores['Z'].argmax(1),native[order+'|stock'][rr].argmax(1)), 'frozen zero')
                for policy in ('F','P'):
                    scores[policy+'pair'] = np.mean([scores[policy+str(seed)] for seed in SEEDS],axis=0)
                pack.update({key+k:v.argmax(1).astype('uint8') for k,v in scores.items()})
    check(bool(pack),'empty shard')
    name = f'{mode}_{idx}'
    np.savez_compressed(name+'.npz',**pack)
    Path(name+'.json').write_text(json.dumps(dict(source=digest('source.sha256'),outputs=digest(name+'.npz'))))


if __name__=='__main__':
    main(sys.argv[1],int(os.environ.get('SLURM_ARRAY_TASK_ID',0)))
