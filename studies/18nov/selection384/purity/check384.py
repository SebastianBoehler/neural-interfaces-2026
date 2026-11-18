"""Label-assisted training-row diagnostic, released only after all deployed paths are scored."""
import hashlib
import json
from pathlib import Path
import os
import sys

import numpy as np

ORIGINAL = Path('/mnt/lustre/work/gehler/gfh527/ni26_selection384_v1_20261118')
sys.path.insert(0,str(ORIGINAL))
from inputs384 import ARMS, CHAIN, COHORT, INPUT, ORDERS, SCHEDULE, SEEDS, banks, check, digest, setup
from calibrate import evaluate
from experiment import interval
from measure import official


def source():
    check(digest(ORIGINAL/'source.sha256')=='d58a4d9b75aa6cca648f59126eb593e51c42f4cc242346a65e0d0d55fe840500',
          'primary changed')
    for line in (ORIGINAL/'source.sha256').read_text().splitlines():
        h,n = line.split(maxsplit=1)
        check(digest(ORIGINAL/n)==h,'primary code changed')
    report = json.loads((ORIGINAL/'result.json').read_text())
    check(report['proof']==digest(ORIGINAL/'before_labels.json'),'primary completeness')
    return setup()


def worker(idx):
    _,cp,cfg,meta,rows,reader = source()
    held = COHORT[idx]
    target = rows[held]
    true = reader(CHAIN/'data/proteus_windows.npz',target,len(meta['person']))
    feat = np.load(SCHEDULE/'features.npy',mmap_mode='r')
    archive = {}
    seeds = {}
    for seed in SEEDS:
        bank = banks(seed,rows)[held]
        card = json.loads((ORIGINAL/f'{seed}_{held}.json').read_text())
        check(card['outputs']==digest(ORIGINAL/f'{seed}_{held}.npz'),'primary outputs differ')
        seeds[seed] = (bank,card)
    with np.load(INPUT/'inputs.npz') as ctl:
        for o in ORDERS:
            rr = np.concatenate([r for r in np.split(ctl[o+'|index'],ctl[o+'|bounds'][1:-1])
                                 if np.array_equal(np.sort(r),target)])
            pos = np.searchsorted(target,rr)
            raw,filt,gold = (ctl[o+'|'+k][rr] for k in ('raw','filtered','expected'))
            actual = true[pos]
            arms = {}
            for seed in SEEDS:
                bank, card = seeds[seed]
                s = evaluate(card['pair'],bank,o,True)[2][pos]
                arms[seed] = s
            purity = {k:[] for k in ('zero','oracle','382S','382D','1382S','1382D')}
            changes = {k:0 for k in purity}
            class Oracle(cp):
                def select(self,y,c):
                    t = len(y)
                    z = super().select(y,c)
                    correct = y==self.previous_labels
                    # Correctness is primary and confidence breaks ties.
                    oracle = super().select(y,correct.astype(float)*2+c)
                    selected = dict(zero=z,oracle=oracle)
                    for seed,s in arms.items():
                        value = s[:t][np.arange(t),y]
                        shuffle = value.copy()
                        for label in np.unique(y):
                            group = np.flatnonzero(y==label)
                            rng = np.random.default_rng(384+10*t+int(label))
                            shuffle[group] = value[rng.permutation(group)]
                        selected[str(seed)+'S'] = super().select(y,.75*c+.25*value)
                        selected[str(seed)+'D'] = super().select(y,.75*c+.25*shuffle)
                    for k,keep in selected.items():
                        check(np.array_equal(np.bincount(y[keep],minlength=3),np.bincount(y[z],minlength=3)),
                              'class quota changed')
                        changes[k] += len(np.setdiff1d(keep,z))
                        if t+cfg['v6_cp_k']>cfg['v6_cp_w0']:
                            purity[k].append(np.mean([correct[keep][y[keep]==cl].mean() for cl in np.unique(y[keep])]))
                    for cl in np.unique(y):
                        check(correct[oracle][y[oracle]==cl].sum()==min(np.sum(correct & (y==cl)),
                                                                       np.sum(y[oracle]==cl)),
                              'correctness ceiling failed')
                    return oracle
                def __del__(self):
                    pass
            model = Oracle(cfg)
            model.feats = feat.__getitem__
            decisions = []
            for i,r in enumerate(rr):
                model.previous_labels = actual[:i]
                decisions.append(model.step(r,raw[i],filt[i],False))
            out = np.stack(decisions)
            check(not model.dead and model.log['err']==0 and np.array_equal(out[:cfg['v6_cp_w0']],gold[:cfg['v6_cp_w0']]),
                  'invalid label diagnostic')
            for seed,(_,card) in seeds.items():
                for arm in ('S','D'):
                    check(changes[str(seed)+arm]==card['traces'][o+'|'+arm]['replaced'],'primary rank trace mismatch')
            archive[o+'|oracle'] = out.argmax(1)[np.argsort(pos)]
            for k,v in purity.items():
                archive[o+'|purity_'+k] = 100*np.mean(v)
    np.savez_compressed(held+'.npz',index=target,**archive)


def collect():
    _,_,_,meta,rows,reader = source()
    metric = official()
    n = len(meta['person'])
    arr = {k:np.empty((4,8)) for k in ('score_zero','score_oracle','purity_zero','purity_oracle',
                                         'purity_382S','purity_382D','purity_1382S','purity_1382D')}
    pins = {}
    for i,p in enumerate(COHORT):
        y = reader(CHAIN/'data/proteus_windows.npz',rows[p],n)
        pins[p] = digest(p+'.npz')
        with np.load(p+'.npz') as z, np.load(ORIGINAL/f'382_{p}.npz') as old:
            check(np.array_equal(z['index'],old['index']) and np.array_equal(z['index'],rows[p]),'rows changed')
            for j,o in enumerate(ORDERS):
                for k,d in [('score_zero',old[o+'|Z']),('score_oracle',z[o+'|oracle'])]:
                    v = metric(y,d,meta,meta['person']==p)
                    check(v['n_cells']==2,'unexpected metric weights')
                    arr[k][i,j] = 100*v['balanced_accuracy_cell_mean']
                for k in arr:
                    if k.startswith('purity_'):
                        arr[k][i,j] = z[o+'|'+k]
    for a,b in [('score_oracle','score_zero'),('purity_oracle','purity_zero')]+[
            ('purity_'+str(seed)+'S',ref) for seed in SEEDS for ref in ('purity_zero','purity_'+str(seed)+'D')]:
        arr[a+'-'+b] = arr[a]-arr[b]
    result = {k:interval(v) for k,v in arr.items()}
    check(abs(result['score_zero']['pp']-81.21561492579204)<1e-9,'stock changed')
    with open('result.json','x') as f:
        json.dump(dict(metrics=result,pins=pins,source=digest('source.sha256'),primary=digest(ORIGINAL/'result.json')),f,indent=2)


if __name__=='__main__':
    if sys.argv[1]=='score':
        collect()
    else:
        worker(int(os.environ['SLURM_ARRAY_TASK_ID']))
