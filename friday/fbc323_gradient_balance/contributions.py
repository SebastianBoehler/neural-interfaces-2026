"""Measure native class-weighted head contributions at each real source update."""
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


HERE=Path(__file__).resolve().parent
KINDS=("hard","balance","mirrored")
MODES=("ordinary","bounded")
ARMS=tuple(f"{k}_{m}" for k in KINDS for m in MODES)
CAP=.10
CHUNK=32
WIDTH=1152
PARAMETERS=347903


def eligibility(flat,streams):
    people=np.asarray([v.rsplit("|",1)[0] for v in streams])
    result=[]
    for rows in flat:
        groups=streams[rows].reshape(-1,CHUNK)
        if len(groups) not in (1,2) or np.any(groups!=groups[:,:1]):
            raise ValueError("native update does not contain original single-session 32-row chunks")
        result.append(len(groups)==2 and people[rows[0]]!=people[rows[CHUNK]])
    return np.asarray(result,dtype=bool)


def gradients(features,logits,labels,weights):
    mass=weights[labels].reshape(2,CHUNK)
    totals=mass.sum(1)
    residual=(logits.softmax(1)-F.one_hot(labels,3)).reshape(2,CHUNK,3)
    local=residual*(mass/totals[:,None])[...,None]
    x=torch.cat((features[:,:WIDTH],torch.ones_like(features[:,:1])),dim=1).reshape(2,CHUNK,WIDTH+1)
    return totals/totals.sum(),local.transpose(1,2)@x


def autograd_probe(task,features,logits,labels,weights,analytic):
    with torch.enable_grad():
        w=task.weight.detach().clone().requires_grad_()
        b=task.bias.detach().clone().requires_grad_()
        twin=F.linear(features,w,b)
        if not torch.equal(twin,logits):
            raise ValueError("cloned affine head differs from the actual source output")
        errors=[]
        for j,r in enumerate((slice(0,CHUNK),slice(CHUNK,2*CHUNK))):
            grad,bias=torch.autograd.grad(F.cross_entropy(twin[r],labels[r],weight=weights),(w,b),
                                          retain_graph=j==0)
            actual=torch.cat((grad[:,:WIDTH],bias[:,None]),dim=1)
            if grad[:,WIDTH:].count_nonzero():
                raise ValueError("cloned probe used discarded fixed-feature coordinates")
            errors.append(float((actual-analytic[j]).abs().max()))
    error=max(errors)
    if not 0<=error<=2e-6:
        raise ValueError(f"analytic versus autograd head gradient differs by {error:.6g}")
    return error


class Survey:
    def __init__(self,model,x,y,weights,flat,eligible,probe=False):
        self.x,self.y,self.weights=x,y,weights
        self.flat,self.eligible=flat,eligible
        self.rows=[]
        self.checked=False
        self.probe=probe
        self.error=None
        self.handles=(model.register_forward_pre_hook(self.check_rows),model.task.register_forward_hook(self.read_head))

    @torch.no_grad()
    def check_rows(self,module,args):
        t=len(self.rows)
        if (self.checked or t>=len(self.flat) or len(args)!=1 or
                not torch.equal(args[0].detach(),self.x[self.flat[t]])):
            raise ValueError("actual native source forward changed fixed update rows or order")
        self.checked=True

    @torch.no_grad()
    def read_head(self,module,args,output):
        t=len(self.rows)
        if not self.checked or len(args)!=1 or args[0].shape!=(len(self.flat[t]),1439):
            raise ValueError("native task affine did not receive the expected source update")
        features=args[0].detach()
        if torch.count_nonzero(features[:,WIDTH:]):
            raise ValueError("native waveform training passed nonzero fixed features")
        if self.eligible[t]:
            labels=self.y[self.flat[t]]
            fractions,g=gradients(features,output.detach(),labels,self.weights)
            if self.probe and self.error is None:
                self.error=autograd_probe(module,features,output.detach(),labels,self.weights,g)
            self.rows.append(torch.cat((fractions,g.flatten(1).norm(dim=1))).double())
        else:
            self.rows.append(output.new_zeros(4,dtype=torch.float64))
        self.checked=False

    def detach(self):
        for handle in self.handles:
            handle.remove()

    def proof(self):
        if self.checked or len(self.rows)!=len(self.flat) or self.probe and self.error is None:
            raise ValueError("actual source trajectory is incomplete")
        values=torch.stack(self.rows).cpu().numpy()
        active=values[self.eligible]
        if (not np.isfinite(values).all() or not len(active) or np.any(active[:,:2]<=CAP) or
                not np.allclose(active[:,:2].sum(1),1.,atol=2e-7,rtol=0) or
                np.any(active[:,2:].sum(1)<=0) or np.any(values[~self.eligible])):
            raise ValueError("invalid per-group class-weighted head contributions")
        values[self.eligible,1]=1-values[self.eligible,0]
        return values


def transfer(reference,eligible):
    a,b,n1,n2=reference[eligible].T
    offsets=np.zeros(len(reference),dtype=np.float64)
    offsets[eligible]=np.clip((b*n2-a*n1)/(n1+n2),-CAP,CAP)
    if not np.any(np.abs(offsets)>1e-3):
        raise ValueError("no active bounded source loss transfer")
    return offsets


def imbalance(trace,eligible,offsets):
    base=trace[eligible,:2]
    d=offsets[eligible,None]*np.asarray([1.,-1.])
    coefficients=base+d
    contribution=coefficients*trace[eligible,2:]
    if np.any(coefficients<=0) or np.max(np.abs(coefficients.sum(1)-1.))>1e-15:
        raise ValueError("balanced or mirrored loss lost positive, exactly unit total mass")
    gap=np.abs(contribution[:,0]-contribution[:,1])
    return float(gap.mean()),float((gap/contribution.sum(1)).mean())
