"""Preserve every PROTEUS optimizer step with a separate motor head optimizer."""
import torch
import torch.nn.functional as F

from auxiliary import check_tail, inspect_gradient, motor
from external import WEIGHT
from source_targets import normalizer


def train(model,head,x,y,mu,sd,weights,flat,fx,fy,tape,seed,detached):
    model.moment_mean.copy_(mu)
    model.moment_sd.copy_(sd)
    layer=normalizer(model)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    private=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.001)
    schedule=torch.optim.lr_scheduler.OneCycleLR(opt,.001,total_steps=len(flat))
    private_schedule=torch.optim.lr_scheduler.OneCycleLR(private,.001,total_steps=len(flat))
    torch.manual_seed(seed)
    model.train()
    head.train()
    source_x=fx[tape[0]]
    proof=check_tail(model,head,source_x)
    maximum=0.
    first=last=None
    external=[]
    for rows,ext in zip(flat,tape,strict=True):
        opt.zero_grad(set_to_none=True)
        private.zero_grad(set_to_none=True)
        ce=F.cross_entropy(model(x[rows]),y[rows],weight=weights)
        native_mean,native_var=layer.running_mean.clone(),layer.running_var.clone()
        extra=F.cross_entropy(motor(model,head,fx[ext],detached),fy[ext])
        if not torch.equal(native_mean,layer.running_mean) or not torch.equal(native_var,layer.running_var):
            raise ValueError("external data altered native SessionBN running buffers")
        if not external:
            proof.update(inspect_gradient(model,head,extra,detached))
        loss=ce+WEIGHT*extra
        if not torch.isfinite(loss):
            raise ValueError("nonfinite mixed-corpus training objective")
        loss.backward()
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        if not grads or any(not g.isfinite().all() for g in grads):
            raise ValueError("nonfinite native waveform gradients")
        maximum=max(maximum,max(float(g.abs().max()) for g in grads))
        opt.step()
        private.step()
        schedule.step()
        private_schedule.step()
        last=float(ce.detach())
        first=last if first is None else first
        external.append(extra.detach())
    model.eval()
    return dict(steps=len(flat),max_gradient=maximum,first_loss=first,last_loss=last,actual_aux=len(external),
                external_mean_loss=float(torch.stack(external).mean()),native_norm_isolated=True,**proof)
