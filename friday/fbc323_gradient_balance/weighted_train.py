"""Preserve the original hard optimizer and apply only the bounded mass transfer."""
import torch
from torch.nn import functional as F

from contributions import CHUNK


def train(model,x,y,mu,sd,weights,flat,offsets,seed):
    model.moment_mean.copy_(mu)
    model.moment_sd.copy_(sd)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    schedule=torch.optim.lr_scheduler.OneCycleLR(opt,.001,total_steps=len(flat))
    torch.manual_seed(seed)
    model.train()
    maximum=0.
    first=last=None
    applied=[]
    for rows,offset in zip(flat,offsets,strict=True):
        opt.zero_grad(set_to_none=True)
        logits,labels=model(x[rows]),y[rows]
        loss=F.cross_entropy(logits,labels,weight=weights)
        if offset:
            a=F.cross_entropy(logits[:CHUNK],labels[:CHUNK],weight=weights)
            b=F.cross_entropy(logits[CHUNK:],labels[CHUNK:],weight=weights)
            loss=loss+float(offset)*(a-b)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite matched weighted source loss")
        loss.backward()
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise ValueError("invalid matched weighted source gradients")
        maximum=max(maximum,max(float(g.abs().max()) for g in grads))
        opt.step()
        schedule.step()
        last=float(loss.detach())
        first=last if first is None else first
        applied.append(float(offset))
    if not maximum>0:
        raise ValueError("no active source gradient")
    model.eval()
    return dict(steps=len(applied),first_loss=first,last_loss=last,max_gradient=maximum),applied
