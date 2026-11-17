"""Full-dimensional finite Gaussian hypothesis banks with explicit view precisions."""
import copy
import itertools
import numpy as np
from scipy.special import logsumexp,softmax

PERMS=np.array(list(itertools.permutations(range(3))))
ETAS=(.001,.01,.1)
CONFIGS={'uniform':('mixture',0.,1.),'original334':('mixture',.25,1.)}
for eta in ETAS:
    for mode in ('static','semantic_translation','translation'):CONFIGS[f'{mode}|{eta}']=(mode,eta,1.)
    for rho in (1.,100.,10000.):CONFIGS[f'mixture|{eta}|{rho}']=('mixture',eta,rho)


def components(model,z,mode,neutral=False,cyclic=False):
    g=model['geometry'] if mode=='mixture' else model['global_geometry'][None]
    b=model['offset'][:1] if mode=='static' else model['offset']
    ba=b@model['q'];br=b-ba@model['q'].T
    means=(g[:,None]+br[None,:,None]).reshape(-1,3,z.shape[1])
    baa=np.broadcast_to(ba[None],(len(g),len(b),2)).reshape(-1,2)
    a=z@model['q'];r=z-a@model['q'].T
    anchor=model['anchor'][[2,0,1]] if cyclic else model['anchor']
    aa=-.25*((a[:,None,None]-baa[None,:,None]-anchor[None,None])**2).sum(-1)
    if neutral:aa=np.broadcast_to(-.25*((a[:,None]-baa[None])**2).sum(-1)[:,:,None],aa.shape)
    rr=-.5*((means*means).sum(-1)[None]-2*(r@means.reshape(-1,z.shape[1]).T).reshape(len(z),-1,3))/(z.shape[1]-2)
    if mode in ('static','semantic_translation'):rr=np.zeros_like(rr)
    perms=PERMS if mode=='mixture' else PERMS[:1]
    return aa[:,:,None]+np.zeros_like(rr[:,:,perms]),rr[:,:,perms],(len(g),len(b),len(perms))


class State:
    def __init__(self,shape):self.shape=shape;self.reset()
    def reset(self):self.evidence=np.zeros(self.shape)
    def step(self,lp):
        self.evidence+=logsumexp(lp,axis=-1)-np.log(3)
        self.evidence-=logsumexp(self.evidence)
        return (np.exp(self.evidence)[...,None]*softmax(lp,axis=-1)).sum((0,1))


def stream(lp):
    s=State(lp.shape[1:3]);p=np.stack([s.step(v) for v in lp])
    if not np.isfinite(p).all() or not np.allclose(p.sum(-1),1,atol=1e-10):raise ValueError('posterior')
    return p,s


def ordered(lp,rows,d,permutation):
    rank=np.empty(len(permutation),int);rank[permutation]=np.arange(len(permutation))
    p=np.zeros((len(rows),3));activity=[]
    for session in dict.fromkeys(d['stream'][rows]):
        ix=np.flatnonzero(d['stream'][rows]==session);ix=ix[np.argsort(rank[rows[ix]])]
        p[ix],s=stream(lp[ix]);activity.append(s.evidence)
    return p,activity


def guards(model,z):
    a,r,shape=components(model,z,'mixture');lp=.01*(a+10000*r)
    p,s=stream(lp);q=State(s.shape);q.step(lp[0]);c=copy.deepcopy(q)
    if not np.array_equal(q.step(lp[1]),c.step(lp[1])):raise ValueError('deepcopy')
    if not np.array_equal(stream(lp[:5])[0],p[:5]):raise ValueError('future')
    q.reset()
    if not np.array_equal(np.stack([q.step(v) for v in lp]),p):raise ValueError('reset')
    stream(lp[::-1])
    if not np.array_equal(stream(lp)[0],p):raise ValueError('A-B-A')
    if not np.allclose(stream(lp[::-1])[1].evidence,s.evidence,atol=1e-9):raise ValueError('setorder')
    if not np.allclose(stream(lp*0)[0],1/3,atol=1e-14):raise ValueError('uniform0')
    an,rn,_=components(model,z,'mixture',neutral=True)
    if not np.allclose(stream(.01*(an+10000*rn))[0],1/3,atol=1e-10):raise ValueError('neutralanchor')
    ac,rc,_=components(model,z,'mixture',cyclic=True)
    if not np.allclose(stream(.01*(ac+10000*rc))[0],p[:,[2,0,1]],atol=1e-9):raise ValueError('cyclic')
    # Independent original334 implementation, pinned immutable source.
    import sys
    sys.path.insert(0,'/mnt/lustre/work/gehler/gfh527/ni26_gen334_v7_20261117')
    from inference import banks,density_stream
    old=banks(model,z,.25)['mixture'];oldp,_=density_stream(model,old,.25,'mixture')
    if not np.allclose(.25*(a+r),old,atol=1e-9) or not np.allclose(stream(.25*(a+r))[0],oldp,atol=1e-9):raise ValueError('334identity')
    # Explicit Gaussian differences vs omitted shared squared norm, one actual window.
    from inference import Session
    h=Session(model,.25,'mixture');x=z[0];aa=x@h.q;rr=x-aa@h.q.T
    direct=-.01/4*((aa[None,None]-h.ba[:,None]-h.anchor[None])**2).sum(-1)[:,None]
    direct=direct-.01*10000/2*((rr[None,None]-h.mean)**2).sum(-1)[:,h.perms]/(len(x)-2)
    if not np.allclose(softmax(direct.reshape(-1)),softmax(lp[0].reshape(-1)),atol=1e-9):raise ValueError('directGaussian')
    return dict(passed=True,original334_max_abs=float(np.abs(stream(.25*(a+r))[0]-oldp).max()),dimensions=z.shape[1])
