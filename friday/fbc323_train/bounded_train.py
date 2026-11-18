"""Apply F13 bounded past moments in the original S1C training chunks."""
import copy

import torch

from bounded_bn import preceding


def prior_windows(value):
    return preceding(value.transpose(0, 1)).transpose(0, 1)


def bounded_training(layer, x, activity, enabled=True):
    m, k = layer.m, layer.k_train
    if not layer.training or not layer.causal or m != 32 or k != 20 or len(x) % m:
        raise ValueError("expected original 32-window causal S1C training chunks")
    axes = tuple(range(2, x.ndim))
    z = x.reshape(len(x) // m, m, *x.shape[1:])
    reduce = tuple(a + 1 for a in axes)
    r1, r2 = z.mean(reduce), (z * z).mean(reduce)
    rm, rv = layer.running_mean.detach(), layer.running_var.detach()
    n = torch.arange(1, m + 1, dtype=x.dtype, device=x.device)[None, :, None]
    mean = (k * rm + r1.cumsum(1)) / (k + n)
    variance = ((k * (rv + rm ** 2) + r2.cumsum(1)) / (k + n) - mean ** 2).clamp_min(0)
    if enabled:
        z64, rm64, rv64 = z.double(), rm.double(), rv.double()
        m1, m2 = z64.mean(reduce), z64.square().mean(reduce)
        with torch.no_grad():
            energy = m2 - 2 * m1 * rm64 + rm64.square()
            w = (4 * rv64.clamp_min(layer.eps) / energy.clamp_min(layer.eps)).clamp(max=1)
            clipped = w < 1
            active = prior_windows(clipped.to(torch.int32)) > 0
            activity.append(torch.count_nonzero(clipped[:, :-1]))
        mass = k + 1 + prior_windows(w)
        center = (k * rm64 + m1 + prior_windows(w * m1)) / mass
        square = k * (rv64 + rm64.square()) + m2 + prior_windows(w * m2)
        spread = (square / mass - center.square()).clamp_min(0)
        mean = torch.where(active, center.to(x.dtype), mean)
        variance = torch.where(active, spread.to(x.dtype), variance)
    with torch.no_grad():
        gm = r1.mean(1)
        gv = (r2.mean(1) - gm ** 2).clamp_min(0)
        layer.running_mean.mul_(1 - layer.momentum).add_(layer.momentum * gm.mean(0))
        layer.running_var.mul_(1 - layer.momentum).add_(layer.momentum * gv.mean(0))
    shape = (len(x), -1) + (1,) * len(axes)
    output = (x - mean.reshape(shape)) / torch.sqrt(variance.reshape(shape) + layer.eps)
    return layer._affine(output, (1, -1) + (1,) * len(axes))


def verify_training(layer, real):
    """Test the pre-affine training activations from actual source EEG."""
    native, zero = (copy.deepcopy(layer).train() for _ in range(2))
    native.causal = zero.causal = True
    a, b = (real.detach().clone().requires_grad_() for _ in range(2))
    p, q = native(a), bounded_training(zero, b, [], enabled=False)
    if not torch.equal(p, q):
        raise ValueError("disabled bounded training changed the original causal normalization")
    p.square().mean().backward()
    q.square().mean().backward()
    if not torch.equal(a.grad, b.grad) or any(not torch.equal(x, y) for x, y in zip(
            native.state_dict().values(), zero.state_dict().values(), strict=True)):
        raise ValueError("disabled bounded training changed source gradients or population EMA")
    original = copy.deepcopy(layer).train()
    altered = copy.deepcopy(layer).train()
    original.causal = altered.causal = True
    probe = real.detach().clone().requires_grad_()
    other = real.detach().clone()
    half = layer.m // 2
    other[half:] = other[half:].flip(0)
    bounded = bounded_training(original, probe, [])
    first = bounded[:half]
    with torch.no_grad():
        prefix = bounded_training(altered, other, [])[:half]
    first.square().mean().backward()
    if (not torch.equal(bounded[::layer.m], p[::layer.m]) or not torch.equal(first, prefix)
            or not torch.isfinite(probe.grad).all()
            or torch.count_nonzero(probe.grad[half:]).item()):
        raise ValueError("bounded training reads future windows or another training chunk")
