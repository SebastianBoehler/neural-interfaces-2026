"""Bound past-window influence in the original S1C evaluation layer."""
import torch


def preceding(value):
    return torch.cat((torch.zeros_like(value[:1]), value[:-1].cumsum(0)))


def activation_probe(layer, x):
    """Find the earliest real active transition for the label-free oracle."""
    centered = x[:-1].double().flatten(2) - layer.running_mean.double()[None, :, None]
    limit = 4 * layer.running_var.double().clamp_min(layer.eps)
    active_rows = torch.nonzero((centered.square().mean(2) > limit).any(1))
    if not len(active_rows):
        return None
    first = int(active_rows[0, 0])
    return x[max(0, first - 8):first + 2].clone()


def normalize(layer, x, statistics, enabled=True):
    if layer.training or layer.k != 20.0 or x.ndim < 3 or not len(x):
        raise ValueError("expected a nonempty S1C evaluation stream and k=20")
    axes = tuple(range(2, x.ndim))
    r1, r2 = x.mean(axes), (x * x).mean(axes)
    rm, rv = layer.running_mean, layer.running_var
    n = torch.arange(1, len(x) + 1, dtype=x.dtype, device=x.device)[:, None]
    total = layer.k + n
    mean = (layer.k * rm + r1.cumsum(0)) / total
    second = layer.k * (rv + rm ** 2) + r2.cumsum(0)
    variance = (second / total - mean ** 2).clamp_min(0)
    shape = (len(x), -1) + (1,) * len(axes)
    if enabled:
        # FP32 subtraction of nearly equal second moments fails on real S1C activations.
        x64, rm64, rv64 = x.double(), rm.double(), rv.double()
        m1 = x64.mean(axes)
        m2 = x64.square().mean(axes)
        energy = m2 - 2 * m1 * rm64 + rm64.square()
        w = (4 * rv64.clamp_min(layer.eps) / energy.clamp_min(layer.eps)).clamp(max=1)
        clipped = w < 1
        active = preceding(clipped.to(torch.int32)) > 0
        mass = layer.k + 1 + preceding(w)
        center = (layer.k * rm64 + m1 + preceding(w * m1)) / mass
        power = layer.k * (rv64 + rm64.square()) + m2 + preceding(w * m2)
        spread = (power / mass - center.square()).clamp_min(0)
        mean = torch.where(active, center.to(x.dtype), mean)
        variance = torch.where(active, spread.to(x.dtype), variance)
        statistics.append(torch.count_nonzero(clipped[:-1]))
    out = (x - mean.view(shape)) / torch.sqrt(variance.view(shape) + layer.eps)
    return layer._affine(out, (1, -1) + (1,) * len(axes))


def verify_normalization(layer, x, active_required):
    """Check real activations against the original and a direct float64 calculation."""
    original = layer(x)
    unweighted = normalize(layer, x, [], enabled=False)
    if not torch.equal(original, unweighted):
        raise ValueError("disabled bounded normalization changed original activations")
    bounded = normalize(layer, x, [])
    x64 = x.double()
    location = layer.running_mean.double()[None, :, None]
    dispersion = layer.running_var.double()[None, :, None]
    flat = x64.flatten(2)
    energy = (flat - location).square().mean(2, keepdim=True).clamp_min(layer.eps)
    w = (4 * dispersion.clamp_min(layer.eps) / energy).clamp(max=1)
    result = []
    for i in range(len(flat)):
        weights = torch.cat((w[:i], torch.ones_like(w[i:i+1])))
        total = layer.k + weights.sum(0)
        mean = (layer.k * location[0] + (flat[:i+1] * weights).mean(2).sum(0)[:, None]) / total
        pop_distance = dispersion[0] + (location[0] - mean).square()
        squared = (flat[:i+1] - mean).square()
        variance = (layer.k * pop_distance + (squared * weights).mean(2).sum(0)[:, None]) / total
        result.append((flat[i] - mean) / torch.sqrt(variance + layer.eps))
    expected = torch.stack(result).reshape(x.shape)
    expected = layer._affine(expected, (1, -1) + (1,) * (x.ndim - 2))
    active = (preceding((w < 1).to(torch.int32)) > 0).reshape(
        len(x), x.shape[1], *((1,) * (x.ndim - 2))).expand_as(x)
    if not torch.equal(bounded[~active], original[~active]):
        raise ValueError("bounded normalization changed activations without clipped history")
    if bool(active.any()) != active_required:
        raise ValueError("the oracle did not reproduce the activation scan")
    if not active_required:
        return 0.0
    active_expected = expected[active]
    active_actual = bounded[active].double()
    torch.testing.assert_close(active_actual, active_expected, rtol=2e-5, atol=2e-5)
    return float((active_actual - active_expected).abs().max())
