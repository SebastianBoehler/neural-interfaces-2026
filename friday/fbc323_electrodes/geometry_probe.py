"""Test the actual zero initialization, source spectral activation and fitted gradients."""
from copy import deepcopy

import torch

from electrodes import ACTIVE, BASE, attach, parameters, trainable
from source_targets import raw_logits


@torch.no_grad()
def install(model, kind, graph, x):
    cpu, gpu = torch.random.get_rng_state(), torch.cuda.get_rng_state()
    probe = deepcopy(model).eval()
    base, logits = probe.representation(x), raw_logits(probe, x)
    attach(probe, kind, graph)
    if not torch.equal(base, probe.representation(x)) or not torch.equal(logits, raw_logits(probe, x)):
        raise ValueError("zero linked electrode features changed the original native logits")
    layer = attach(model, kind, graph)
    if (trainable(model) != parameters(kind) or not torch.equal(cpu, torch.random.get_rng_state()) or
            not torch.equal(gpu, torch.cuda.get_rng_state())):
        raise ValueError("zero initialization changed RNG or actual trainable work")
    return layer


@torch.no_grad()
def diagnostics(model, layer, sample, alt):
    x = layer.spectral(sample)
    changed, rms = layer.lift(x, layer.neighbors)
    swapped, _ = layer.lift(x, torch.as_tensor(alt, device=x.device))
    identity_delta = float((changed - x).abs().max())
    graph_delta = float((changed - swapped).abs().max())
    bound = float(((changed - x) / rms).abs().max())
    weight = float(layer.theta.abs().max())
    count = int(layer.theta.count_nonzero())
    if (not changed.isfinite().all() or min(identity_delta, graph_delta, weight) <= ACTIVE or
            not 0 < bound <= .250002 or count <= 0):
        raise ValueError("trained spatial gate is inactive, nonfinite or beyond its analytic bound")
    return dict(identity_delta=identity_delta, graph_delta=graph_delta, ratio_to_rms=bound,
                weight_max=weight, nonzero_weights=count)


def fit(model, layer, args, ref):
    from encoder312 import fingerprint
    import model322

    q, weights = [], []
    readout = model.task.weight.register_hook(lambda g: q.append(g[:, -287:].count_nonzero()))
    graph = None if layer is None else layer.theta.register_hook(lambda g: weights.append(g.abs().max()))
    try:
        values = model322.train(model, *args)
    finally:
        readout.remove()
        if graph is not None:
            graph.remove()
    n = ref["steps"]
    state = fingerprint(model)
    if values["steps"] != n or len(q) != n or torch.stack(q).sum().item() or state == ref["initial"]:
        raise ValueError("native fit work or exact 287 zero-feature gradients differ")
    if layer is None:
        if state != ref["state"] or trainable(model) != BASE:
            raise ValueError("hard trained state differs from the exact F16 anchor")
    else:
        grad = torch.stack(weights)
        if len(grad) != n or not grad.isfinite().all() or not (grad > 0).all() or grad.max() <= ACTIVE:
            raise ValueError("learned electrode weights did not participate in all updates")
        values["graph_gradient"] = float(grad.max())
    return dict(values, state=state)
