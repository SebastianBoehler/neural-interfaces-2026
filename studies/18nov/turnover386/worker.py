"""Frozen cap250 replay on one person and half the delivery orders."""
import json
import os
import sys
from pathlib import Path
import numpy as np
from selector import CORE, GEOM, exchange

SCHEDULE = Path('/mnt/lustre/work/gehler/gfh527/ni26_refit375_v1_20261117')
sys.path.insert(0,str(SCHEDULE))
from trajectory import INPUT,check,digest,prepare


def setup():
    manifest = SCHEDULE/'source.sha256'
    check(digest(manifest)=='289a7f6d7540ddaa4db1f42db5a1f67a323fcc2626057e94a5bdd76694404b1c','parent identity')
    for line in manifest.read_text().splitlines():
        h,n = line.split(maxsplit=1)
        check(digest(SCHEDULE/n)==h,'parent input changed')
    assets,_,_ = prepare()
    cls,cfg = assets.get_stage()
    return assets,cls,cfg


def replay(cls,cfg,rows,feat,geometry,raw,filt,arm,stop=None):
    trace = [0,0,0]
    previous = np.array([],dtype=int)
    previous_t = 0
    class CP(cls):
        def select(self,y,c):
            nonlocal previous,previous_t
            stock = super().select(y,c)
            active = exchange(y,c,stock,arm,geometry[:len(y)])
            check(np.array_equal(np.bincount(y[stock]),np.bincount(y[active])),'class budget')
            count = sum(min((y==k).sum()-(y[stock]==k).sum(),int((y[stock]==k).sum()/8+.5)) for k in np.unique(y))
            replaced = len(np.setdiff1d(active,stock))
            check(replaced==(0 if arm=='Z' else count),'exchange budget')
            trace[0] += replaced
            trace[1] += len(np.setxor1d(previous,active[active<previous_t]))
            trace[2] += 1
            previous,previous_t = active,len(y)
            return active
        def __del__(self):
            pass
    cp = CP(cfg)
    cp.feats = feat.__getitem__
    p = np.stack([cp.step(r,raw[r],filt[r],i==stop) for i,r in enumerate(rows)])
    horizon = len(rows) if stop is None else stop
    check(not cp.dead and not cp.log['err'] and cp.fused==max(0,horizon-90)
          and cp.refits==trace[2]==len(range(20,horizon,20)),'stage activity')
    return p,trace


def main(mode,shard):
    smoke = mode=='smoke'
    arms = CORE+GEOM if smoke else CORE if mode=='core' else GEOM
    assets,cls,cfg = setup()
    featpath = SCHEDULE/'features.npy'
    if smoke:
        check(digest(featpath)=='fcbd25192a1e1a030c0905f2cc795c21e1984cbb7ff9c1fbd4b5653e21ec9103','features pin')
    feat = np.load(featpath,mmap_mode='r')
    with np.load(assets.CHAIN/'data/proteus_windows.npz') as z:
        people = np.array([p.rsplit('/',1)[1] for p in z['subject']])
        streams = np.char.add(np.char.add(z['subject'],'/'),z['session'])
    payload = {}
    with np.load(INPUT/'inputs.npz') as z,np.load(INPUT/'outputs.npz') as native,np.load(assets.CHAIN/'runs/shuffle_perms.npz') as perms:
        for order in (z['orders'][:1] if smoke else z['orders'][shard%2::2]):
            raw,filt,gold,idx,bounds = [z[order+'|'+k] for k in ('raw','filtered','expected','index','bounds')]
            perm = perms[f'{order[:-1]}|{order[-1]}|all_loso']
            for sid,rows in enumerate(np.split(idx,bounds[1:-1])):
                if people[rows[0]]!=sorted(set(people))[shard//2]:
                    continue
                check(np.array_equal(rows,perm[streams[perm]==streams[rows[0]]]),'canonical shuffled session')
                check(np.array_equal(gold[rows].argmax(1),native[order+'|stock'][rows].argmax(1)),'native endpoint')
                geometry = np.log(np.maximum(np.diagonal(feat[rows],axis1=2,axis2=3),1e-12)).reshape(len(rows),-1)
                key = f'{order}|{sid}|'
                payload[key+'rows'] = rows
                for arm in arms:
                    p,trace = replay(cls,cfg,rows,feat,geometry,raw,filt,arm)
                    if arm=='Z':
                        check(np.array_equal(p,gold[rows]),'stock drift')
                    check(np.array_equal(p[:90],gold[rows][:90]),'early intervention')
                    if smoke:
                        q,_ = replay(cls,cfg,rows[:131],feat,geometry[:131],raw,filt,arm)
                        stop,_ = replay(cls,cfg,rows[:131],feat,geometry[:131],raw,filt,arm,63)
                        check(np.array_equal(q,p[:131]) and np.array_equal(stop,filt[rows[:131]]),'prefix or recognition')
                    payload[key+arm] = p.argmax(1).astype('uint8')
                    payload[key+arm+'_trace'] = trace
    name = f'{mode}_{shard}'
    check(bool(payload),'empty shard')
    np.savez_compressed(name+'.npz',**payload)
    Path(name+'.json').write_text(json.dumps(dict(outputs=digest(name+'.npz'),source=digest('source.sha256'))))


if __name__=='__main__':
    main(sys.argv[1],int(os.environ.get('SLURM_ARRAY_TASK_ID',0)))
