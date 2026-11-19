"""Prove the initialized zero gate and diagnose trained source mechanisms."""
from copy import deepcopy

import torch

from modulation import ACTIVE, BASE, BANDS, DILATIONS, attach, architecture, trainable
from source_targets import raw_logits


@torch.no_grad()
def install(model, kind, sample):
    cpu, gpu = torch.random.get_rng_state(), torch.cuda.get_rng_state()
    probe = deepcopy(model).eval()
    rep, score = probe.representation(sample), raw_logits(probe, sample)
    attach(probe, kind)
    if (not torch.equal(rep, probe.representation(sample)) or not torch.equal(score, raw_logits(probe, sample))):
        raise ValueError("zero learned modulation did not exactly preserve native representation and logits")
    layer = attach(model, kind)
    if (not torch.equal(cpu, torch.random.get_rng_state()) or not torch.equal(gpu, torch.cuda.get_rng_state()) or
            trainable(model) != architecture(kind)["parameters"]):
        raise ValueError("matched gate construction changed RNG or trainable work")
    return layer


@torch.no_grad()
def diagnostics(model, layer, sample, kind):
    inputs = []
    hook = layer.register_forward_pre_hook(lambda _, args: inputs.append(args[0]))
    try:
        model.representation(sample)
    finally:
        hook.remove()
    if len(inputs) != 1:
        raise ValueError("real pre-pool input not observed")
    x = inputs[0]
    modulated, gain = layer.waveform(x, layer.dilations)
    other = DILATIONS["single" if kind == "multi" else "multi"]
    swapped, _ = layer.waveform(x, other)
    actual = layer.pool(modulated)
    identity_delta = float((actual - layer.pool(x)).abs().max())
    dilation_delta = float((actual - layer.pool(swapped)).abs().max())
    shared = layer.bands.reshape(-1, BANDS, BANDS)
    diagonal = torch.eye(BANDS, device=x.device, dtype=torch.bool)
    cross_band = float(shared[:, ~diagonal].abs().max())
    edge_tap = float(torch.cat([w[..., (0, 1, 3, 4)].flatten() for w in layer.taps]).abs().max())
    if (not gain.isfinite().all() or gain.min() < .75 or gain.max() > 1.25 or
            min(identity_delta, dilation_delta, cross_band, edge_tap) <= ACTIVE):
        raise ValueError("modulation did not learn meaningful cross-band and temporal effects")
    return dict(identity_delta=identity_delta, dilation_delta=dilation_delta, cross_band=cross_band,
                edge_tap=edge_tap, gain_min=float(gain.min()), gain_max=float(gain.max()))


def fit(model, layer, args, ref):
    from encoder312 import fingerprint
    import model322

    q, bands, temporal = [], [], []
    hooks = [model.task.weight.register_hook(lambda g: q.append(g[:, -287:].count_nonzero()))]
    if layer is not None:
        hooks.append(layer.bands.register_hook(lambda g: bands.append(g.abs().max())))
        hooks.extend(w.register_hook(lambda g: temporal.append(g.abs().max())) for w in layer.taps)
    try:
        stats = model322.train(model, *args)
    finally:
        for hook in hooks:
            hook.remove()
    n = ref["steps"]
    state = fingerprint(model)
    if stats["steps"] != n or len(q) != n or torch.stack(q).sum().item() or state == ref["initial"]:
        raise ValueError("native source update work or zero-feature guard failed")
    if layer is None:
        if state != ref["state"] or trainable(model) != BASE:
            raise ValueError("hard source trained state differs from exact F16")
    else:
        grad = torch.stack(bands + temporal)
        if len(bands) != n or len(temporal) != 3 * n or not grad.isfinite().all() or not (grad[n:] > 0).all():
            raise ValueError("learned gate did not participate in all fitted steps")
        band_grad = float(grad[:n].max())
        tap_grad = float(grad[n:].max())
        if min(band_grad, tap_grad) <= ACTIVE:
            raise ValueError("learned cross-band and temporal gradients stayed inactive")
        stats.update(band_grad=band_grad, tap_grad=tap_grad)
    return dict(stats, state=state)
