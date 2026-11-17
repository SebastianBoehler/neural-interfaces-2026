"""Estimate shrinkage from per-window activation energies across source sessions."""
import copy
import numpy as np
import torch


def components(model, layer, x, streams):
    energies = []
    def collect(module, args):
        a = args[0].double()
        shape = (1, -1) + (1,)*(a.ndim-2)
        energies.append((a-module.running_mean.reshape(shape)).square().flatten(2).mean(-1).cpu().numpy())
    hook = layer.register_forward_pre_hook(collect)
    try:
        with torch.no_grad():
            for batch in x.split(32):
                model(batch)
    finally:
        hook.remove()
    e = np.concatenate(energies)
    assert len(e) == len(streams) and np.isfinite(e).all()
    within, between = [], []
    v0 = layer.running_var.double().cpu().numpy()
    for session in np.unique(streams):
        values = e[streams == session]
        assert len(values)>1
        variance = values.var(0, ddof=1)
        within.append(variance)
        between.append((values.mean(0)-v0)**2 - variance / len(values))
    w = np.mean(within,0)
    t = np.maximum(0,np.mean(between,0))
    assert (w>0).all() and np.isfinite(w+t).all() and (t>0).any()
    return w,t


def configure(model, layer_type, w, t):
    child = copy.deepcopy(model)
    layers = [m for m in child.body.modules() if isinstance(m,layer_type)]
    assert len(layers) == 1
    layer = layers[0]
    # Stable convex combination includes the T=0 limit as a population-only channel.
    device=layer.running_mean.device
    w = torch.as_tensor(w,device=device,dtype=torch.float32)
    t = torch.as_tensor(t,device=device,dtype=torch.float32)
    assert w.shape == t.shape == layer.running_var.shape
    active=t>0
    k=torch.where(active,w/t,0.)
    assert torch.isfinite(k).all()
    def forward(x):
        dims = tuple(range(2,x.ndim))
        n = torch.arange(1,len(x)+1,dtype=x.dtype,device=x.device)[:,None]
        mean = (k*layer.running_mean + x.mean(dims).cumsum(0))/(k+n)
        m2 = (k*(layer.running_var+layer.running_mean.square()) + x.square().mean(dims).cumsum(0))/(k+n)
        var = (m2-mean.square()).clamp_min(0)
        var=torch.where(active,var,layer.running_var)
        shape=(len(x),-1)+(1,)*len(dims)
        fixed=(1,-1)+(1,)*len(dims)
        return layer._affine((x-layer.running_mean.reshape(fixed))/torch.sqrt(var.reshape(shape)+layer.eps),fixed)
    layer.forward = forward
    return child
