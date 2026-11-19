"""Verify exact no-op initialization and active causal recurrent representations."""
from copy import deepcopy

import torch

from readout import ACTIVE, BASE, EXTRA, TOL, attach, gate_hash, order, trainable
from source_targets import raw_logits


@torch.no_grad()
def activation(model, layer, sample):
    values = []
    hook = layer.register_forward_pre_hook(lambda _, args:values.append(args[0]))
    try:
        model.representation(sample)
    finally:
        hook.remove()
    if len(values) != 1:
        raise ValueError("native coarse pooling path did not execute")
    return values[0]


@torch.no_grad()
def prefix(layer, tokens):
    sequence = tokens[:8].index_select(1, layer.permutation)
    a, _ = layer.recurrent(sequence)
    b, _ = layer.recurrent(sequence[:, :7])
    layer.recurrent(sequence.flip(0))
    reset, _ = layer.recurrent(sequence[:, :7])
    drift = float((a[:, :7] - b).abs().max())
    if not 0 <= drift <= TOL or not torch.equal(b, reset):
        raise ValueError("GRU latent prefix changed with future tokens or interleaving")
    return drift


@torch.no_grad()
def install(model, kind, seed, sample):
    before = torch.random.get_rng_state(), torch.cuda.get_rng_state()
    p = deepcopy(model).eval()
    z, y = p.representation(sample), raw_logits(p,sample)
    trial = attach(p,kind,seed)
    if not torch.equal(z,p.representation(sample)) or not torch.equal(y,raw_logits(p,sample)):
        raise ValueError("zero residual was not exactly native S1C")
    proof = {}
    if trial is not None:
        x = activation(p,trial,sample)
        proof.update(initial_gate=gate_hash(trial), initial_prefix=prefix(trial,trial.tokens(x)))
    layer = attach(model,kind,seed)
    if (trainable(model) != BASE + EXTRA*(layer is not None) or
            not torch.equal(before[0],torch.random.get_rng_state()) or
            not torch.equal(before[1],torch.cuda.get_rng_state()) or
            (layer is not None and gate_hash(layer)!=proof["initial_gate"])):
        raise ValueError("GRU setup changed caller RNG, initialization or parameter count")
    return layer,proof


@torch.no_grad()
def diagnostics(model, layer, sample, kind):
    x = activation(model,layer,sample)
    tokens = layer.tokens(x)
    z = layer.tail(tokens,layer.permutation)
    alt = order("scrambled" if kind=="chronological" else "chronological")
    swap = layer.tail(tokens,torch.as_tensor(alt,device=z.device))
    residual, displacement = float(z.abs().max()), float((z-swap).abs().max())
    if not 0 < residual <= .25 or min(residual, displacement) <= ACTIVE or not z.isfinite().all():
        raise ValueError("the fitted recurrent time order or residual is inactive")
    return dict(residual=residual, order_delta=displacement, recurrent_prefix=prefix(layer,tokens),
                final_gate=gate_hash(layer))


def fit(model, layer, args, base):
    from encoder312 import fingerprint
    import model322

    q, grads = [], {}
    hooks=[model.task.weight.register_hook(lambda g:q.append(g[:,-287:].count_nonzero()))]
    if layer is not None:
        for name,p in (("norm",layer.norm.weight),("project",layer.project.weight),
                       ("recurrent",layer.recurrent.weight_hh_l0),("emit",layer.emit.weight)):
            history=grads[name]=[]
            hooks.append(p.register_hook(lambda g,h=history:h.append(g.abs().max())))
    try:
        stats=model322.train(model,*args)
    finally:
        for hook in hooks:
            hook.remove()
    n=base["steps"]
    state=fingerprint(model)
    if (stats["steps"]!=n or len(q)!=n or torch.stack(q).sum().item() or state==base["initial"] or
            (layer is None and state!=base["state"])):
        raise ValueError("native hard control, source work or zero-feature gradients differ")
    for name, traces in grads.items():
        measured=torch.stack(traces)
        value=float(measured.max())
        if len(traces)!=n or not measured.isfinite().all() or not value>1e-8:
            raise ValueError(f"recurrent source path did not train {name}")
        stats[f"{name}_gradient"]=value
    return dict(stats,state=state)
