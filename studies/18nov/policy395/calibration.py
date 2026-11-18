"""Learn a regularized inverse of the causal score marginal's local sensitivity."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

BASIS=np.array([[1/np.sqrt(2),1/np.sqrt(6)],[-1/np.sqrt(2),1/np.sqrt(6)],[0.,-2/np.sqrt(6)]])
EYE=np.eye(2)


def sufficient(prob):
    q=prob@BASIS
    # Per-window d softmax / d logits projected into the two identifiable class directions.
    j=np.einsum('nk,ki,kj->nij',prob,BASIS,BASIS)-q[:,:,None]*q[:,None,:]
    stats=np.concatenate((q,j.reshape(-1,4),np.ones((len(q),1))),1)
    return np.concatenate((np.zeros_like(stats[:1]),stats.cumsum(0)[:-1]))


class Policy(nn.Module):
    def __init__(self,mean):
        super().__init__()
        self.mean=mean
        self.gain=nn.Parameter(torch.eye(2,dtype=torch.double)*np.arctanh(.5))
        self.diag=nn.Parameter(torch.full((2,),np.log(np.sqrt(20/3)),dtype=torch.double))
        self.cross=nn.Parameter(torch.zeros((),dtype=torch.double))
        self.register_buffer('basis',torch.as_tensor(BASIS))
        self.register_buffer('eye',torch.eye(2,dtype=torch.double))

    def forward(self,stats):
        n=stats[:,-1]
        h=stats[:,2:6].reshape(-1,2,2)
        if self.mean:
            h=n[:,None,None]*self.eye/3
        p=self.diag.exp()
        lower=torch.stack((p[0],p[0]*0,self.cross,p[1])).reshape(2,2)
        system=h+lower@lower.T+1e-6*self.eye
        direction=torch.linalg.solve(system,stats[:,:2,None]).squeeze(-1)
        change=(direction@(2*self.gain.tanh()).T)@self.basis.T
        return 1.5*(change/1.5).tanh()


def training(meta,bank,labels,people):
    features=[]; logits=[]; gold=[]; owner=[]; weights=[]
    for who,order,stream,rows,prob in bank:
        features.append(sufficient(prob))
        logits.append(np.log(prob))
        gold.append(labels[rows])
        owner.extend([who]*len(rows))
        iface=meta['interface'][rows]
        weight=np.zeros(len(rows))
        for c in range(3):
            for i in np.unique(iface):
                k=(labels[rows]==c)&(iface==i)
                assert k.any()
                weight[k]=len(k)/(6*k.sum())
        weights.append(weight)
    arr=[np.concatenate(v) for v in (features,logits,gold,weights)]
    assert (arr[2]>=0).all() and set(owner)==set(people)
    return [torch.from_numpy(v) for v in arr],np.array(owner)


def fit(arr,owner,seed,mean,smoke):
    states,logits,gold,weights=arr
    rng=np.random.default_rng(39500+seed)
    model=Policy(mean)
    opt=torch.optim.Adam(model.parameters(),lr=.01)
    groups=[np.flatnonzero(owner==p) for p in sorted(set(owner))]
    losses=[]; trace=[]; grad=0.
    before=[p.detach().clone() for p in model.parameters()]
    for _ in range(4 if smoke else 400):
        # Equal representation for every source person on each query update.
        rows=np.concatenate([rng.choice(g,8,replace=False) for g in groups])
        trace.append(rows)
        opt.zero_grad()
        correction=model(states[rows])
        loss=(F.cross_entropy(logits[rows]-correction,gold[rows],reduction='none')*weights[rows]).mean()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all()
            grad=max(grad,float(p.grad.abs().max()))
        opt.step()
        losses.append(loss.item())
    assert grad>0 and all(not torch.equal(a,b) for a,b in zip(before,model.parameters()))
    return model,dict(gradient=grad,initial_ce=losses[0],last_ce=losses[-1],
                      params=[p.detach().tolist() for p in model.parameters()]),np.stack(trace)


@torch.no_grad()
def forward(prob,stats,model):
    logits=torch.as_tensor(np.log(prob))
    change=model(torch.as_tensor(stats))
    result=(logits-change).softmax(1).numpy()
    assert np.isfinite(result).all() and np.allclose(result.sum(1),1)
    return result
