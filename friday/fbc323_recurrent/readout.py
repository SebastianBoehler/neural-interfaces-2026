"""Keep the native coarse features and read a bounded residual from fine causal tokens."""
import hashlib
from pathlib import Path

from braindecode.modules.stats import LogVarLayer
import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
KINDS = ("hard", "chronological", "scrambled")
MODES = ("ordinary", "bounded")
ARMS = tuple(f"{k}_{m}" for k in KINDS for m in MODES)
CHANNELS, TOKENS, WIDTH, SAMPLES = 288, 16, 48, 30
SHAPE = (CHANNELS, 4, 120)
BASE, EXTRA, MAX_RESERVED = 347903, 85008, 6_000_000_000
MAX_RATIO, ACTIVE, TOL = 4., 1e-5, 2e-5


def order(kind):
    native = np.arange(TOKENS)
    if kind == "chronological":
        return native
    if kind == "scrambled":
        shuffled = native.copy()
        shuffled[1:-1] = np.random.default_rng(29029).permutation(native[1:-1])
        if np.array_equal(native, shuffled):
            raise ValueError("order-neutral comparison did not disturb chronology")
        return shuffled
    raise ValueError("invalid recurrent mode")


def gate_hash(layer):
    digest = hashlib.sha256()
    for name, value in sorted(layer.named_parameters()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class Readout(nn.Module):
    def __init__(self, pool, kind, seed, device):
        super().__init__()
        self.pool = pool
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(29000 + seed)
            self.norm = nn.LayerNorm(CHANNELS)
            self.project = nn.Linear(CHANNELS, WIDTH)
            self.recurrent = nn.GRU(WIDTH, WIDTH, batch_first=True)
            self.emit = nn.Linear(WIDTH, CHANNELS * 4)
        with torch.no_grad():
            self.emit.weight.zero_()
            self.emit.bias.zero_()
        self.register_buffer("permutation", torch.as_tensor(order(kind)))
        self.to(device)

    def tokens(self, x):
        if x.shape[1:] != SHAPE:
            raise ValueError("native prepool geometry changed")
        t = x.reshape(len(x), CHANNELS, TOKENS, SAMPLES).var(-1).clamp(1e-6, 1e6).log()
        return torch.nn.functional.gelu(self.project(self.norm(t.transpose(1, 2))))

    def tail(self, tokens, permutation):
        hidden, _ = self.recurrent(tokens.index_select(1, permutation))
        return .25 * self.emit(hidden[:, -1]).tanh()

    def forward(self, x):
        pooled = self.pool(x)
        delta = self.tail(self.tokens(x), self.permutation)
        return pooled + delta.reshape_as(pooled)


def attach(model, kind, seed):
    body = model.body
    pool = body.temporal_layer
    origin = LogVarLayer(dim=3)
    if (model.skip or model.dim != CHANNELS * 4 or model.task.in_features != CHANNELS * 4 + 287 or
            model.task.out_features != 3 or body.n_times_padded != TOKENS*SAMPLES or body.stride_factor != 4 or
            type(pool) is not type(origin) or any(getattr(pool, f) != getattr(origin, f)
                for f in ("stat_fn", "dim", "keepdim", "clamp_range", "apply_log"))):
        raise ValueError("original zero-feature S1C coarse pool or readout differs")
    if kind == "hard":
        return None
    layer = Readout(pool, kind, seed, model.task.weight.device)
    body.temporal_layer = layer
    return layer


def trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
