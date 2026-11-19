"""Insert one learned within-block covariance coefficient per spatial output."""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from braindecode.modules.stats import LogVarLayer

from anchors import row_sha


HERE = Path(__file__).resolve().parent
KINDS = ("hard", "lag", "scrambled")
MODES = ("ordinary", "bounded")
ARMS = tuple(f"{kind}_{mode}" for kind in KINDS for mode in MODES)
BLOCK, CHANNELS, BLOCKS = 120, 288, 4
LIMIT = 0.25
BASE_PARAMS = 347903
SHAPE = CHANNELS, BLOCKS, BLOCK
ANALYTIC_TOL = 2e-10
ACTIVE_TOL = 1e-5
TIMING_CAP = 2.


def permutation(kind):
    n = np.arange(BLOCK)
    lag = (n // 12) * 12 + (n + 6) % 12
    if kind == "lag":
        pairs = lag
    elif kind == "scrambled":
        relabel = np.random.default_rng(22022).permutation(BLOCK)
        pairs = np.empty_like(relabel)
        pairs[relabel] = relabel[lag]
    else:
        raise ValueError(f"unknown temporal pairing: {kind}")
    spacing = abs(n - pairs)
    if (not np.array_equal(pairs[pairs], n) or (n == pairs).any() or
            (kind == "lag" and not (spacing == 6).all()) or
            (kind == "scrambled" and (spacing <= 6).mean() >= .2)):
        raise ValueError(f"invalid temporal pairing: {kind}")
    return pairs


def pair_sha(kind):
    return row_sha(permutation(kind)) if kind != "hard" else None


def mix(x, paired, alpha):
    return (x + alpha * paired) * torch.rsqrt(1 + alpha.square())


class LagPool(nn.Module):
    def __init__(self, pool, kind, device):
        super().__init__()
        self.pool = pool
        self.theta = nn.Parameter(torch.zeros(CHANNELS, device=device))
        self.register_buffer("pairs", torch.as_tensor(permutation(kind), device=device))

    def coefficients(self):
        return LIMIT * self.theta.tanh()

    def forward(self, x):
        if x.shape[1:] != SHAPE:
            raise ValueError("unexpected S1C pre-pool shape")
        paired = x.index_select(-1, self.pairs)
        return self.pool(mix(x, paired, self.coefficients().view(1, -1, 1, 1)))


def attach(model, kind):
    body = model.body
    pool = body.temporal_layer
    reference_pool = LogVarLayer(dim=3)
    if (model.skip or model.dim != CHANNELS * BLOCKS or
            model.task.in_features != model.dim + 287 or model.task.out_features != 3 or
            body.stride_factor != BLOCKS or body.n_times_padded != BLOCK * BLOCKS or
            type(pool) is not type(reference_pool) or any(
                getattr(pool, key) != getattr(reference_pool, key)
                for key in ("stat_fn", "dim", "keepdim", "clamp_range", "apply_log"))):
        raise ValueError("original zero-feature S1C dimensions or log-variance pool differ")
    if kind == "hard":
        return None
    layer = LagPool(body.temporal_layer, kind, model.task.weight.device)
    body.temporal_layer = layer
    return layer


def trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
