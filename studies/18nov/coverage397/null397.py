"""Specificity control with random active/baseline swaps in both source corpora."""
from itertools import combinations
import json
from pathlib import Path
import sys
import numpy as np
import features as source
from produce import fit
from common import DATA, ROOT, features, fields, auc


def run(k,smoke):
    part=[]
    for i,task in enumerate(('A','W')):
        paths=sorted((ROOT/'data/shin2017b').glob('windows_*.npz')) if i==0 else [ROOT/'data/shin2018/windows.npz']
        with np.load(paths[0]) as z:
            n=set(z['ch_names'])-set(z['interpolated'])
        with np.load(DATA[k]) as z:
            n&=set(z['ch_names'])-set(z['interpolated'] if 'interpolated' in z.files else [])
        source.CH=tuple(sorted(n))
        source.PAIRS=np.array(list(combinations(range(len(n)),2)))
        pairs,_,_=source.data393(task,paths)
        w=fit(pairs,np.random.default_rng([397,i]),3 if smoke else 128)
        differences=np.concatenate([np.concatenate(v) for v in pairs.values()])
        sd=(differences@w.T).std(0)
        assert np.all(sd>0) and not np.allclose(w[0],w[1])
        with np.load(DATA[k]) as z:x=features(z,source.CH)
        scores=(x@w.T)/sd
        assert np.allclose(scores[[4,2]],(x[[4,2]]@w.T)/sd)
        part.append(scores)
    a,w=part
    v=np.column_stack([w[:,0]+a[:,0],w-a])
    with np.load('../ni26_coverage397_v1_20261118/shin'+k+'.npz') as orig:
        assert np.max(np.abs(v[:,1]-orig['wide']))<1e-8
    if not smoke:np.save('scores'+k+'.npy',v)


def score():
    out={};rng=np.random.default_rng(397)
    orig=json.loads(Path('../ni26_coverage397_v1_20261118/result.json').read_text())
    for k in DATA:
        v=np.load('scores'+k+'.npy')
        with np.load(DATA[k]) as z:
            p,c,g=fields(k,z);y=z['y']
        people=[]
        for pp in sorted(set(p)):
            cells=[]
            for cc in sorted(set(c[p==pp])):
                runs=[]
                for gg in sorted(set(g[c==cc])):
                    mask=g==gg
                    runs.append(auc(y[mask],v[mask]))
                cells.append(np.mean(runs,0))
            people.append(np.mean(cells,0))
        people=np.array(people);m=people.mean(0)
        assert abs(m[1]-orig[k]['shin']['arms']['wide']['mean'][0])<1e-8
        dif=people[:,1]-people[:,0]
        boot=dif[rng.integers(len(dif),size=(10000,len(dif)))].mean(1)
        out[k]=dict(observed=m[1],generic=m[0],difference=dif.mean(),ci=np.quantile(boot,[.1,.9]).tolist(),
                    null90=np.quantile(m[2:],.9),null_mean=m[2:].mean(),
                    exceedances=int((m[2:]>=m[1]).sum()),null=m[2:].tolist())
    Path('result.json').write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':
    if sys.argv[1]=='score':score()
    else:run(sys.argv[2],sys.argv[1]=='smoke')
