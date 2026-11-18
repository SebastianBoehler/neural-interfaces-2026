"""Re-run stock causal personalization with a single intervention in the row selector."""
import json
import os
from pathlib import Path
import sys
import numpy as np

from calibrate import evaluate,fit
from inputs384 import ARMS, COHORT, INPUT, ORDERS, SCHEDULE, SEEDS, banks, check, digest, setup


def replay(parent,cfg,feature,raw,filt,wave,arm,recognition=None):
    decisions,selected = [],[]
    class Rank(parent):
        def select(self,y,c):
            original = super().select(y,c)
            observed = np.asarray(self.wave[:len(y)])
            check(len(y)==len(self.wave)-1, 'selector included current window')
            if arm=='Z':
                active = original
            else:
                value = observed[np.arange(len(y)),y].copy()
                if arm=='D':
                    for label in np.unique(y):
                        mask = np.flatnonzero(y==label)
                        rng = np.random.default_rng(384+10*len(y)+int(label))
                        value[mask] = value[rng.permutation(mask)]
                active = super().select(y,.75*c+.25*value)
            check(np.array_equal(np.bincount(y[original],minlength=3),np.bincount(y[active],minlength=3)),
                  'fit allocation per pseudo-class changed')
            selected.append(int(len(np.setdiff1d(active,original))))
            return active
        def __del__(self):
            pass
    model = Rank(cfg)
    model.wave = []
    model.feats = feature.__getitem__
    for i,(r,f,v) in enumerate(zip(raw,filt,wave,strict=True)):
        model.wave.append(v)
        decisions.append(model.step(i,r,f,i==recognition))
    check(not model.dead and not model.log['err'], 'CP entered its error path')
    expected_refits = sum(cfg['v6_cp_t0']<=i and (i-cfg['v6_cp_t0'])%cfg['v6_cp_k']==0
                          for i in range(len(raw) if recognition is None else recognition))
    check(model.refits==expected_refits and len(selected)==expected_refits,'refit count changed')
    return np.stack(decisions),dict(refits=expected_refits,fused=model.fused,
                                   replaced=sum(selected),events_changed=sum(n>0 for n in selected))


def run(smoke,idx):
    seed,held = SEEDS[idx//len(COHORT)],COHORT[idx%len(COHORT)]
    assets,cp,cfg,meta,rows,reader = setup()
    bank = banks(seed,rows)
    pair = fit(held,rows,meta,bank,reader,True)
    single = fit(held,rows,meta,bank,reader,False)
    features = np.load(SCHEDULE/'features.npy',mmap_mode='r')
    target = rows[held]
    output,traces = {},{}
    with np.load(INPUT/'inputs.npz') as ctl,np.load(INPUT/'outputs.npz') as native:
        for o in (ORDERS[:1] if smoke else ORDERS):
            index = ctl[o+'|index']
            bounds = ctl[o+'|bounds']
            sessions = [r for r in np.split(index,bounds[1:-1]) if np.array_equal(np.sort(r),target)]
            check(len(sessions)==1,'held ownership/order changed')
            rr = sessions[0]
            pos = np.searchsorted(target,rr)
            # Independently verify canonical row → delivery index against pinned shuffle archive.
            with np.load(assets.CHAIN/'runs/shuffle_perms.npz') as z:
                perm = z[f'{o[:-1]}|{o[-1]}|all_loso']
                check(np.array_equal(rr,perm[np.isin(perm,target)]),'pinned delivery order mismatch')
            raw = ctl[o+'|raw'][rr]
            filt = ctl[o+'|filtered'][rr]
            gold = ctl[o+'|expected'][rr]
            r,t,s = evaluate(pair,bank[held],o,True)
            k = evaluate(single,bank[held],o,False)[2]
            check(np.isfinite([r,t,s,k]).all() and np.max(abs(s-r))>1e-7,'learned calibration inactive')
            wave = {'Z':r,'R':r,'T':t,'S':s,'D':s,'K':k}
            check(np.array_equal(gold.argmax(1),bank[held]['frozen',o][pos].argmax(1))
                  and np.array_equal(gold.argmax(1),native[o+'|stock'][rr].argmax(1)), 'frozen/native actor identity')
            feat = features[rr]
            for arm in ARMS:
                p,trace = replay(cp,cfg,feat,raw,filt,wave[arm][pos],arm)
                if arm=='Z':
                    check(np.array_equal(p,gold),'replayed original scores differ')
                check(np.array_equal(p[:cfg['v6_cp_w0']],gold[:cfg['v6_cp_w0']])
                      and trace['fused']==len(rr)-cfg['v6_cp_w0'], 'unauthorized early/late change')
                if smoke and arm in ('S','D'):
                    prefix = 129
                    short,_ = replay(cp,cfg,feat[:prefix],raw[:prefix],filt[:prefix],wave[arm][pos[:prefix]],arm)
                    check(np.array_equal(p[:prefix],short), 'prefix/reset differs')
                    stop,active = replay(cp,cfg,feat[:prefix],raw[:prefix],filt[:prefix],wave[arm][pos[:prefix]],arm,63)
                    check(np.array_equal(stop,filt[:prefix]) and active['fused']==0,'recognition stop failed')
                output[o+'|'+arm] = p.argmax(1)[np.argsort(pos)].astype('uint8')
                traces[o+'|'+arm] = trace
    check(sum(v['replaced'] for k,v in traces.items() if k.endswith('|S'))>0,'no fit selection changes')
    name = 'smoke' if smoke else f'{seed}_{held}'
    np.savez_compressed(name+'.npz',index=target,**output)
    proof = dict(seed=seed,held=held,labels_opened=False,source=digest('source.sha256'),
                 pair=pair,single=single,traces=traces,outputs=digest(name+'.npz'))
    with open(name+'.json','x') as f:
        json.dump(proof,f,indent=2)
    print(name,'complete',flush=True)


if __name__=='__main__':
    run(sys.argv[1]=='smoke',int(os.environ.get('SLURM_ARRAY_TASK_ID',0)))
