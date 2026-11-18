import sys
import numpy as np
from common import *
from source_labels import read_training_labels


def prepare(k):
    with np.load(DATA[k]) as z:
        f=features(z,N19)
        direct=features(z,C4)
        assert np.allclose(f[:,positions(C4)],direct,rtol=1e-9,atol=1e-9)
    np.save(k+'.npy',f)


def fold(k,i):
    x=np.load(k+'.npy')
    with np.load(DATA[k]) as z:p,_,g=fields(k,z)
    held='all' if i<0 else sorted(set(p))[i]
    ix=np.arange(len(p)) if i<0 else np.flatnonzero(p!=held)
    y=read_training_labels(DATA[k],ix,len(p))
    c,mu,mid,sd=fit_cov(x[ix],p[ix],g[ix],y)
    other='P' if k=='S' else 'S'
    target=other if i<0 else k
    with np.load(DATA[target]) as z:pp,_,_=fields(target,z)
    test=np.arange(len(pp)) if i<0 else np.flatnonzero(pp==held)
    xt=np.load(target+'.npy')[test]
    rng=np.random.default_rng(397)
    subsets=[('C4',C4)]+[('R'+str(j),rng.choice(N19,4,replace=False)) for j in range(8)]+[('N19E',N19)]
    out={'index':test,'train':ix}
    for arm,subset in subsets:
        pos=positions(subset)
        e,v=eigh(c[np.ix_(pos,pos)])
        def add(name,dim,reg):
            w,l,df=vector(e,v,mu[pos],width=dim,lam=reg)
            if reg is None:assert abs(df-18)<1e-7
            out[name]=((xt[:,pos]-mid[pos])/sd[pos])@w
            out[name+'_df']=df
            return l
        lam=add(arm,None,None)
        if arm=='C4':common=lam
        if arm=='N19E':
            add('N19',None,common)
            add('N19P',36,None)
    np.savez(f'{k}_{i}.npz',**out)


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare(sys.argv[2])
    else:fold(sys.argv[2],int(sys.argv[3]))
