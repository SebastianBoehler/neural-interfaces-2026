"""Read separable local and cross-band modulation before the native log-variance pool."""
from pathlib import Path

from braindecode.modules.stats import LogVarLayer
import torch
from torch import nn
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
KINDS = ("hard", "single", "multi")
MODES = ("ordinary", "bounded")
ARMS = tuple(f"{kind}_{mode}" for kind in KINDS for mode in MODES)
BANDS, SPATIAL, BLOCKS, LENGTH = 9, 32, 4, 120
C = BANDS * SPATIAL
WIDTHS = (99, 99, 90)
BASE = 347903
EXTRA = C * BANDS + C * 5
MAX_RESERVED = 8_000_000_000
MAX_RATIO = 3.
ACTIVE = 1e-5
DILATIONS = dict(hard=(), single=(1, 1, 1), multi=(1, 4, 12))


def trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ModulatedPool(nn.Module):
    def __init__(self, pool, kind, device):
        super().__init__()
        self.pool = pool
        self.dilations = DILATIONS[kind]
        weight = torch.eye(BANDS, device=device).repeat(SPATIAL, 1).unsqueeze(-1)
        self.bands = nn.Parameter(weight)
        self.taps = nn.ParameterList(nn.Parameter(torch.zeros(w, 1, 5, device=device)) for w in WIDTHS)

    def waveform(self, x, dilations):
        if x.shape[1:] != (C, BLOCKS, LENGTH) or len(dilations) != len(WIDTHS):
            raise ValueError("unexpected native pre-pool shape or temporal scales")
        batch = len(x)
        grouped = x.reshape(batch, BANDS, SPATIAL, -1).transpose(1, 2).reshape(batch, C, -1)
        energy = F.conv1d(grouped.square(), self.bands, groups=SPATIAL)
        parts = energy.split(WIDTHS, dim=1)
        filtered = [F.conv1d(value.contiguous(), w, padding=2 * d, dilation=d, groups=len(w))
                    for value, w, d in zip(parts, self.taps, dilations, strict=True)]
        gates = torch.cat(filtered, dim=1).reshape(batch, SPATIAL, BANDS, BLOCKS, LENGTH).transpose(1, 2)
        gain = 1 + .25 * gates.tanh().reshape_as(x)
        return x * gain, gain

    def forward(self, x):
        value, _ = self.waveform(x, self.dilations)
        return self.pool(value)


def attach(model, kind):
    body = model.body
    pool = body.temporal_layer
    reference = LogVarLayer(dim=3)
    if (model.skip or model.dim != C * BLOCKS or model.task.in_features != C * BLOCKS + 287 or
            model.task.out_features != 3 or body.n_bands != BANDS or body.n_filters_spat != SPATIAL or
            body.n_times_padded != BLOCKS * LENGTH or body.stride_factor != BLOCKS or
            type(pool) is not type(reference) or any(getattr(pool, k) != getattr(reference, k)
                    for k in ("stat_fn", "dim", "keepdim", "clamp_range", "apply_log"))):
        raise ValueError("unexpected native S1C representation or task head")
    if kind == "hard":
        return None
    layer = ModulatedPool(pool, kind, model.task.weight.device)
    body.temporal_layer = layer
    return layer


def architecture(kind):
    return dict(dilations=list(DILATIONS[kind]), parameters=BASE if kind == "hard" else BASE + EXTRA)
