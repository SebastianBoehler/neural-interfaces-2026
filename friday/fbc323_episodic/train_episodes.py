"""Train native S1C and differentiate through a source-only temporary residual."""
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from episodes import WEIGHT, analytic, hypothetical, unit


def head(model,z):
    return model.task(torch.cat((z,z.new_zeros((len(z),287))),1))


def train(model,x,y,mu,sd,weights,flat,doses,seed,kind):
    model.moment_mean.copy_(mu)
    model.moment_sd.copy_(sd)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    scheduler=torch.optim.lr_scheduler.OneCycleLR(opt,.001,total_steps=len(flat))
    torch.manual_seed(seed)
    model.train()
    mx = 0.
    first = last = None
    proof = {}
    aux, delta = [], []
    for r,roles in zip(flat,doses,strict=True):
        opt.zero_grad(set_to_none=True)
        z=model.representation(x[r])
        scores=head(model,z)
        loss=F.cross_entropy(scores,y[r],weight=weights)
        if roles is not None:
            support,query=roles
            warm=unit(z)
            shifted,correction=hypothetical(warm,scores,y[r],support,query,kind)
            extra=F.cross_entropy(shifted.flatten(0,1),y[r][query].flatten())
            if not aux:
                error=analytic(warm,scores,y[r],support,query,kind)
                raw_grad,=torch.autograd.grad(extra,z,retain_graph=True)
                proof=dict(analytic_error=error,support_gradient=float(raw_grad[support].abs().max()),
                           query_gradient=float(raw_grad[query].abs().max()))
                if min(proof["support_gradient"],proof["query_gradient"])<=1e-8 or not np.isfinite(list(proof.values())).all():
                    raise ValueError("hypothetical source update did not differentiate through both exposure roles")
            loss=loss+WEIGHT*extra
            aux.append(extra.detach())
            delta.append(correction.detach().abs().max())
        if not torch.isfinite(loss):
            raise ValueError("nonfinite episodic source objective")
        loss.backward()
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        if not grads or any(not g.isfinite().all() for g in grads):
            raise ValueError("invalid native source gradient")
        mx=max(mx,max(float(g.abs().max()) for g in grads))
        opt.step()
        scheduler.step()
        last=float(loss.detach())
        first=last if first is None else first
    model.eval()
    aux_mean=float(torch.stack(aux).mean())
    move=float(torch.stack(delta).max())
    if not 0<aux_mean or not 1e-5<move<=4.00002:
        raise ValueError("temporary head was inactive or unbounded")
    return dict(steps=len(flat),max_gradient=mx,first_loss=first,last_loss=last,actual_episodes=len(aux),
                temporary_move=move,auxiliary_mean=aux_mean,**proof)


@torch.no_grad()
def diagnostic(model,x,y,flat,doses,picked):
    copy=deepcopy(model).train()
    losses=[]
    for step in picked:
        r=flat[step]
        z=copy.representation(x[r])
        support,query=doses[step]
        current,_=hypothetical(unit(z),head(copy,z),y[r],support,query,"cross")
        losses.append(F.cross_entropy(current.flatten(0,1),y[r][query].flatten()))
    value=float(torch.stack(losses).mean())
    if not np.isfinite(value) or not value>0:
        raise ValueError("fixed real-source cross-query diagnostic did not execute")
    return value
