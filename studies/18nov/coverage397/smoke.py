from common import *
from source_labels import read_training_labels

with np.load(DATA['S']) as z:
    x=z['X'][:120]
    names=z['ch_names'].tolist()
    p,_,g=fields('S',z)
r=np.arange(120)
assert len(set(p[r]))==1
y=read_training_labels(DATA['S'],r,len(p))
f=extract(x[:,[names.index(c) for c in N19]][:8])
f4=extract(x[:,[names.index(c) for c in C4]])
assert np.allclose(f[:,positions(C4)],f4[:8]) and np.allclose(f[:1],extract(x[:1,[names.index(c) for c in N19]]))
c,mu,m,sd=fit_cov(f4,p[r],g[r],y)
e,v=eigh(c)
w,l,df=vector(e,v,mu)
assert np.isfinite(w).all() and abs(df-18)<1e-7 and l>0 and len(w)==36
