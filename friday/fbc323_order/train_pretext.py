"""Apply the training-only auxiliary without changing the native S1C state schema."""
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from order_pretext import WEIGHT,invariance,patches


def capture(model,x):
    result=[]
    hook=model.body.temporal_layer.register_forward_pre_hook(lambda _,inp:result.append(inp[0]))
    try:
        z=model.representation(x)
    finally:
        hook.remove()
    if len(result)!=1:
        raise ValueError("auxiliary did not observe the actual unmodified waveform body")
    return model.task(torch.cat((z,z.new_zeros((len(z),287))),1)),result[0]


def train(model,head,x,y,mu,sd,weights,flat,doses,seed,kind):
    model.moment_mean.copy_(mu)
    model.moment_sd.copy_(sd)
    params=list(model.parameters())+list(head.parameters())
    opt=torch.optim.AdamW(params,lr=.001,weight_decay=.001)
    lr=torch.optim.lr_scheduler.OneCycleLR(opt,.001,total_steps=len(flat))
    torch.manual_seed(seed)
    model.train()
    maximum=0.
    first=last=None
    proof={}
    errors=[]
    for rows,(slot,signs) in zip(flat,doses,strict=True):
        opt.zero_grad(set_to_none=True)
        pred,features=capture(model,x[rows])
        crop=patches(features,slot)
        score=head(crop)
        # For the other orientation the score and binary target both reverse.
        extra=F.softplus(-score if kind=="ordered" else -score*signs).mean()
        if not errors:
            spatial=model.body.spatial_conv[0].parametrizations.weight.original
            body,aux=torch.autograd.grad(extra,(spatial,head.project.weight),retain_graph=True)
            proof=dict(invariance_drift=invariance(head,crop),body_gradient=float(body.abs().max()),
                       head_gradient=float(aux.abs().max()))
            if min(proof["body_gradient"],proof["head_gradient"])<=1e-8:
                raise ValueError("auxiliary gradients did not reach trainable source representation and head")
        errors.append(extra.detach())
        loss=F.cross_entropy(pred,y[rows],weight=weights)+WEIGHT*extra
        if not torch.isfinite(loss):
            raise ValueError("invalid matched raw/source auxiliary loss")
        loss.backward()
        grads=[p.grad for p in params if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise ValueError("nonfinite or inactive real gradient")
        maximum=max(maximum,max(float(g.abs().max()) for g in grads))
        opt.step()
        lr.step()
        last=float(loss.detach())
        first=last if first is None else first
    model.eval()
    return dict(steps=len(flat),max_gradient=maximum,first_loss=first,last_loss=last,
                aux_steps=len(errors),pretext_loss=float(torch.stack(errors).mean()),**proof)


@torch.no_grad()
def diagnostic(model,head,x,flat,doses):
    probe=deepcopy(model).train()
    outputs=[]
    for rows,(slot,_) in zip(flat[:6],doses[:6],strict=True):
        _,value=capture(probe,x[rows])
        outputs.append(head(patches(value,slot)))
    score=torch.cat(outputs)
    if not torch.isfinite(score).all() or float(score.abs().max())<=1e-6:
        raise ValueError("held-source chronology diagnostic lacked active scores")
    return dict(true_source_loss=float(F.softplus(-score).mean()),
                source_order_accuracy=float((score>0).float().mean()),
                maximum_pretext_score=float(score.abs().max()))
