"""Real source waveform diagnostics before and after fitting the temporal mixer."""
from statistics import median
from time import perf_counter

import torch

from source_targets import raw_logits
from temporal import ACTIVE_TOL, ANALYTIC_TOL, LIMIT, mix, permutation


@torch.no_grad()
def timing(model, sample):
    for _ in range(3):
        raw_logits(model, sample)
    times = []
    for _ in range(7):
        torch.cuda.synchronize()
        start = perf_counter()
        raw_logits(model, sample)
        torch.cuda.synchronize()
        times.append((perf_counter() - start) * 1000)
    return median(times)


@torch.no_grad()
def activation(model, layer, sample):
    seen = []
    handle = layer.register_forward_pre_hook(lambda module, data: seen.append(data[0]))
    try:
        model.representation(sample)
    finally:
        handle.remove()
    if len(seen) != 1:
        raise ValueError("missing actual pre-pool source activation")
    return seen[0]


@torch.no_grad()
def analytic(model, layer, sample):
    x = activation(model, layer, sample).double()
    paired = x.index_select(-1, layer.pairs)
    alpha = x.new_tensor(.125)
    xc = x - x.mean(-1, keepdim=True)
    pc = paired - paired.mean(-1, keepdim=True)
    covariance = (xc * pc).sum(-1) / (x.shape[-1] - 1)
    expected = x.var(-1) + 2 * alpha / (1 + alpha.square()) * covariance
    error = float((mix(x, paired, alpha).var(-1) - expected).abs().max())
    if not 0 <= error <= ANALYTIC_TOL:
        raise ValueError(f"actual source covariance identity differs: {error}")
    return error


@torch.no_grad()
def effects(model, layer, sample, kind):
    x = activation(model, layer, sample)
    if not x.isfinite().all():
        raise ValueError("nonfinite source pre-pool activation")
    beta = layer.coefficients()
    actual = layer(x)
    other = torch.as_tensor(permutation("scrambled" if kind == "lag" else "lag"), device=x.device)
    swapped = layer.pool(mix(x, x.index_select(-1, other), beta.view(1, -1, 1, 1)))
    zero_delta = float((actual - layer.pool(x)).abs().max())
    swap_delta = float((actual - swapped).abs().max())
    maximum = float(beta.abs().max())
    if not ACTIVE_TOL < min(zero_delta, swap_delta) or not 0 < maximum <= LIMIT:
        raise ValueError("trained temporal coefficients or alternate pairing had no effect")
    return dict(coefficient_max=maximum, coefficient_mean=float(beta.abs().mean()),
                nonzero_coefficients=int(beta.count_nonzero()), zero_delta=zero_delta, swap_delta=swap_delta)
