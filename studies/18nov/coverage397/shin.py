"""Calibrate both §393 class directions using only their own source corpus."""
from pathlib import Path
import sys
import numpy as np
import features as features393
import produce as produce393
from common import C4, DATA, ROOT, features

def main(target):
    banks={};results={}
    for task,paths in [('A',sorted((ROOT/'data/shin2017b').glob('windows_*.npz'))),('W',[ROOT/'data/shin2018/windows.npz'])]:
        with np.load(paths[0]) as z:
            native=set(z['ch_names'])-set(z['interpolated'])
        with np.load(DATA[target]) as z:
            n=set(z['ch_names'])-set(z['interpolated'] if 'interpolated' in z.files else [])
        shared=tuple(sorted(native&n))
        assert len(shared)==(6 if task=='A' or target=='S' else 16)
        for width,ch in [('4',C4),('wide',shared)]:
            # Reuse the §393 source matching and fit with its channel list and pair inventory.
            from itertools import combinations
            features393.CH=ch
            features393.PAIRS=np.array(list(combinations(range(len(ch)),2)))
            sample,rows,_=features393.data393(task,paths)
            def fit(bank):
                return produce393.fit(bank,np.random.default_rng(393),0)[0]
            w=fit(sample)
            diffs=np.concatenate([np.concatenate(v)@w for v in sample.values()])
            sd=np.std(diffs)
            assert sd>1e-6
            per={}
            for p in sample:
                w0=fit({u:v for u,v in sample.items() if u!=p})
                per[p]=np.mean([produce393.auc(y,f@w0) for person,y,f in rows if person==p])
            with np.load(DATA[target]) as z: x=features(z,ch)
            raw=x@w
            if width=='4':banks[task+'raw']=raw
            banks[task+width]=raw/sd
            results[task+width]=np.mean(list(per.values()))*100
    out={'raw':banks['Wraw']-banks['Araw']}
    for a,w,name in [('4','4','C4'),('wide','wide','wide'),('wide','4','Aexpand'),('4','wide','Wexpand')]:
        out[name]=banks['W'+w]-banks['A'+a]
    for a in ('4','wide'):
        out['W'+a]=banks['W'+a]
        out['minusA'+a]=-banks['A'+a]
    out.update({'source_'+k:v for k,v in results.items()})
    np.savez('shin'+target+'.npz',**out)
if __name__=='__main__':main(sys.argv[1])
