"""Exact finite latent-hypothesis Gaussian prediction using prefix log evidence."""
import itertools
import numpy as np
from scipy.special import logsumexp,softmax

ARMS=('static','translation','mixture','hard','cyclic','anchor_off')
PERMS=np.asarray(list(itertools.permutations(range(3))),dtype=int)


class Session:
    def __init__(self,model,eta,arm):
        if arm not in ARMS or eta not in (.25,1.,4.):raise ValueError('unregistered arm/precision')
        self.model,self.eta,self.arm=model,float(eta),arm
        g=model['global_geometry'][None] if arm in ('static','translation') else model['geometry']
        b=model['offset'][:1] if arm=='static' else model['offset']
        self.q=model['q'];self.anchor=model['anchor'][[2,0,1]] if arm=='cyclic' else model['anchor']
        ba=b@self.q;br=b-ba@self.q.T
        self.mean=(g[:,None]+br[None,:,None]).reshape(-1,3,g.shape[-1])
        self.mean2=(self.mean*self.mean).sum(-1)
        self.ba=np.broadcast_to(ba[None],(len(g),len(b),2)).reshape(-1,2)
        self.perms=PERMS[:1] if arm in ('static','translation') else PERMS
        self.shape=(len(self.mean),len(self.perms))
        self.reset()

    def reset(self):
        self.evidence=np.zeros(self.shape,dtype=np.float64);self.n=0

    def logdensity(self,x):
        a=x@self.q;r=x-a@self.q.T
        # Drop the shared ||r||² term: common to every h/y, so it cancels exactly.
        residual=-.5*self.eta*(self.mean2-2*np.einsum('hcd,d->hc',self.mean,r))/(len(x)-2)
        anchor=-.5*self.eta*((a[None,None]-self.ba[:,None]-self.anchor[None])**2).sum(-1)/2
        if self.arm=='anchor_off':anchor=np.broadcast_to(-.25*self.eta*((a[None]-self.ba)**2).sum(-1)[:,None],anchor.shape)
        return anchor[:,None]+residual[:,self.perms]

    def step(self,x):
        return self.step_density(self.logdensity(x))

    def step_density(self,lp):
        self.evidence+=logsumexp(lp,axis=-1)-np.log(3.)
        self.evidence-=logsumexp(self.evidence)
        joint=softmax(lp,axis=-1)
        if self.arm=='hard':out=joint.reshape(-1,3)[self.evidence.argmax()]
        else:out=np.sum(np.exp(self.evidence)[...,None]*joint,axis=(0,1))
        self.n+=1
        if not np.isfinite(out).all() or not np.isclose(out.sum(),1,atol=1e-10):raise ValueError('badposterior')
        return out


def stream(model,z,eta,arm):
    state=Session(model,eta,arm)
    return np.stack([state.step(x) for x in z]),state


def banks(model,z,eta):
    result={}
    for arm in ('static','translation','mixture','cyclic','anchor_off'):
        state=Session(model,eta,arm);a=z@state.q;r=z-a@state.q.T
        residual=-.5*eta*(state.mean2[None]-2*(r@state.mean.reshape(-1,z.shape[1]).T).reshape(len(z),-1,3))/(z.shape[1]-2)
        anchor=-.5*eta*((a[:,None,None]-state.ba[None,:,None]-state.anchor[None,None])**2).sum(-1)/2
        if arm=='anchor_off':anchor=np.broadcast_to(-.25*eta*((a[:,None]-state.ba[None])**2).sum(-1)[:,:,None],anchor.shape)
        result[arm]=anchor[:,:,None]+residual[:,:,state.perms]
    result['hard']=result['mixture']
    return result


def density_stream(model,values,eta,arm):
    state=object.__new__(Session);state.model=model;state.eta=eta;state.arm=arm
    state.shape=values.shape[1:3];state.reset()
    return np.stack([state.step_density(lp) for lp in values]),state
