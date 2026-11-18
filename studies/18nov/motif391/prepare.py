"""Build immutable row-local sufficient statistics without task labels."""
import sys
import numpy as np
import score379 as base
from tokens import PAIRS, counts, features

path=base.BASE/'data/proteus_windows.npz'
assert base.sha(path)==base.DATA_SHA
with np.load(path) as data:
    names=data['ch_names'].tolist()
    x=data['X']
indices=np.array([[names.index(a),names.index(b)] for a,b in PAIRS])
sample=np.array([x[0,indices[0,0]]-x[0,indices[0,1]]],dtype=np.float64)
for lag in (1,3,8):
    for a,b in zip(counts(sample,lag),counts(np.arcsinh(sample),lag)):
        assert np.array_equal(a,b)
smoke=sys.argv[1]=='smoke'
rows=np.arange(17) if smoke else np.arange(len(x))
p,c=zip(*(features(x[i],indices) for i in rows))
p,c=np.stack(p),np.stack(c)
assert x.shape==(2655,41,480) and c.shape==(len(rows),2,36,6,6)
assert np.isfinite(p).all() and np.any(c[:,0]!=c[:,1])
for i in (0,len(rows)-1,5):
    pb,cb=features(x[rows[i]],indices)
    assert np.array_equal(p[i],pb) and np.array_equal(c[i],cb)
np.savez('smoke.npz' if smoke else 'features.npz',p=p,c=c,index=rows)
