import csv
import gzip
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq

ROOT=Path('/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026')
C4=('Cz','P3','P4','Pz')
N19=tuple('F3 Fz F4 FC3 FCz FC4 C3 Cz C4 CP3 CPz CP4 P5 P3 P1 Pz P2 P4 P6'.split())
BANDS=((2,4),(4,8),(8,13),(13,20),(20,30),(30,40))
DATA={'S':ROOT/'data/scherer2015/windows.npz','P':ROOT/'data/proteus_windows.npz'}


def extract(x):
    ij=np.array(list(combinations(range(x.shape[1]),2)))
    d=x[:,ij[:,0]].astype(float)-x[:,ij[:,1]]
    d-=d.mean(-1,keepdims=True)
    ps=np.abs(np.fft.rfft(d*np.hanning(480),axis=-1))**2
    hz=np.fft.rfftfreq(480,1/120)
    mass=np.stack([ps[:,:,(hz>=lo)&(hz<hi)].sum(-1) for lo,hi in BANDS],axis=-1)
    return np.log((mass+1e-12)/(mass.sum(-1,keepdims=True)+6e-12)).reshape(len(x),-1)


def features(z,names):
    ch=z['ch_names'].tolist()
    assert set(names)<=set(ch)-set(z['interpolated'] if 'interpolated' in z.files else [])
    x=z['X'][:,[ch.index(c) for c in names]]
    return np.concatenate([extract(x[i:i+8]) for i in range(0,len(x),8)])


def fields(kind,z):
    p,s,r,t=[z[k] for k in ('subject','session','run','interface')]
    if kind=='S':
        with gzip.open(ROOT/'runs/diag/external_semantics351/scherer_rows.tsv.gz','rt') as f:
            obs=list(csv.DictReader(f,delimiter='\t'))
        assert len(obs)==len(p) and all(v['person']==p[i].split('/')[-1] and v['session']==s[i] and
                                          int(v['row'])==i for i,v in enumerate(obs))
        r=np.array([v['run'] for v in obs])
    cell=np.array(['|'.join(a) for a in zip(p,s,t)])
    run=np.array(['|'.join(a) for a in zip(cell,r.astype(str))])
    return p,cell,run


def positions(subset):
    lookup={frozenset(pair):i for i,pair in enumerate(combinations(N19,2))}
    ids=[lookup[frozenset(pair)] for pair in combinations(subset,2)]
    return (6*np.array(ids)[:,None]+np.arange(6)).ravel()


def vector(eig,V,mu,*,width=None,lam=None):
    if width is not None:eig,V=eig[-width:],V[:,-width:]
    assert np.sum(eig>1e-8)>18
    eig=np.maximum(eig,0)
    if lam is None:lam=brentq(lambda l:(eig/(eig+l)).sum()-18,1e-9,1e6)
    df=float((eig/(eig+lam)).sum())
    return V@((V.T@mu)/(eig+lam)),lam,df


def fit_cov(x,p,g,y):
    res=[];weights=[];mean=[];mid=[]
    for person in sorted(set(p)):
        groups=np.unique(g[p==person]);diff=[];cent=[]
        for name in groups:
            a=x[(g==name)&(y==2)];b=x[(g==name)&(y==1)]
            assert len(a) and len(b)
            ma,mb=a.mean(0),b.mean(0)
            diff.append(ma-mb);cent.append((ma+mb)/2)
            for q,m in ((a,ma),(b,mb)):
                res.append(q-m)
                weights.extend([1/(len(np.unique(p))*len(groups)*2*len(q))]*len(q))
        mean.append(np.mean(diff,0));mid.append(np.mean(cent,0))
    r=np.concatenate(res)*np.sqrt(weights)[:,None]
    cov=r.T@r
    sd=np.sqrt(np.diag(cov))
    assert np.all(sd>1e-8)
    return cov/sd[:,None]/sd[None,:],np.mean(mean,0)/sd,np.mean(mid,0),sd


def auc(y,v):
    d=v[y==2][:,None]-v[y==1][None,:]
    return 100*((d>0)+.5*(d==0)).mean(axis=(0,1))
