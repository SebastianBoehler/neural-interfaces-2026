"""Apply the donor tape between pooled waveforms and the existing raw S1C head."""
from types import MethodType

import numpy as np
import torch
from torch.nn import functional as F


MASS = .25


def head(model, data):
    return model.task(torch.cat((data, data.new_zeros((len(data), 287))), 1))


def combine(data, queries, donors):
    return data.index_copy(0, queries, torch.lerp(data[queries], data[donors], MASS))


def probe(model, x, y, queries, donors):
    with torch.no_grad():
        native = model.representation(x).detach()
    latent = native.clone().requires_grad_(True)
    edited = combine(latent, queries, donors)
    kept = torch.ones(len(x), dtype=torch.bool, device=x.device)
    kept[queries] = False
    if not torch.equal(edited[kept], latent[kept]) or not (edited[queries] != latent[queries]).any():
        raise ValueError("interpolation did not act only at the selected waveform vectors")
    ix, dx = queries[:1], donors[:1]
    loss = F.cross_entropy(head(model, edited[ix]), y[ix])
    grad = torch.autograd.grad(loss, latent)[0]
    a, b = grad[ix], grad[dx]
    torch.testing.assert_close(a / 3., b, rtol=2e-6, atol=2e-7)
    if not torch.count_nonzero(a) or not torch.count_nonzero(b):
        raise ValueError("a real anchor or donor waveform vector received no gradient")
    return dict(anchor_gradient=float(a.norm()), donor_gradient=float(b.norm()))


class LatentMix:
    def __init__(self, bounds, slots, order, device):
        def take(start, end, layout):
            return torch.as_tensor(slots[start:end].astype(np.int64)[:, layout].reshape(-1), device=device)

        self.plan = [(take(i, j, slice(None)), take(i, j, order)) for i, j in zip(bounds[:-1], bounds[1:], strict=True)]
        self.calls = 0
        self.mixed = 0

    def install(self, model):
        def forward(actor, x):
            if not actor.training or self.calls >= len(self.plan):
                raise ValueError("the registered latent mix feed was called out of schedule")
            q, d = self.plan[self.calls]
            self.calls += 1
            self.mixed += q.numel()
            z = actor.representation(x)
            return head(actor, combine(z, q, d))
        return MethodType(forward, model)


@torch.no_grad()
def mechanism(model, x, stream, quartet):
    if model.training or quartet.shape != (64, 4):
        raise ValueError("source mechanism requires 64 fixed unmixed evaluation quartets")
    needed = np.unique(quartet)
    z = torch.zeros((len(x), model.dim), device=x.device)
    for name in np.unique(stream[needed]):
        rows = np.flatnonzero(stream == name)
        z[rows] = model.representation(x[rows])
    a, b, c, d = quartet.T
    z = F.normalize(z, dim=1)
    within = 1. - .5 * ((z[a] * z[b]).sum(1) + (z[c] * z[d]).sum(1))
    cross = 1. - .5 * ((z[a] * z[c]).sum(1) + (z[b] * z[d]).sum(1))
    if not torch.isfinite(within).all() or not torch.isfinite(cross).all():
        raise ValueError("source-only cosine diagnostic is not finite")
    return dict(within_cosine_distance=float(within.mean()), across_cosine_distance=float(cross.mean()),
                across_minus_within_cosine=float((cross - within).mean()), mechanism_quartets=64)
