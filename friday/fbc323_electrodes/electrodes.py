"""Augment native filter-bank signals with bounded signed electrode contrasts."""
import torch
from torch import nn

from topology import DEGREE, HERE, SENSORS


KINDS = ("hard", "physical", "scrambled")
MODES = ("ordinary", "bounded")
ARMS = tuple(f"{kind}_{mode}" for kind in KINDS for mode in MODES)
BANDS, SAMPLES = 9, 480
BASE = 347903
EXTRA = BANDS * SENSORS * DEGREE
MAX_RESERVED = 8_000_000_000
MAX_RATIO = 4.
ACTIVE = 1e-5


class ElectrodeLift(nn.Module):
    def __init__(self, spectral, graph, device):
        super().__init__()
        self.spectral = spectral
        self.theta = nn.Parameter(torch.zeros(BANDS, SENSORS, DEGREE, device=device))
        self.register_buffer("neighbors", torch.as_tensor(graph, device=device))

    def lift(self, x, graph):
        if x.shape[1:] != (BANDS, SENSORS, SAMPLES) or graph.shape != (SENSORS, DEGREE):
            raise ValueError("unexpected filterbank activation or electrode topology")
        power = x.square().mean(-1, keepdim=True) + 1e-4
        rms = power.sqrt()
        z = torch.zeros_like(x)
        weights = self.theta.tanh()
        for k, indices in enumerate(graph.unbind(1)):
            scaled = (x - x.index_select(2, indices)) * torch.rsqrt(power + power.index_select(2, indices))
            z = z + scaled.tanh() * weights[:, :, k][None, :, :, None]
        return x + (.25 / DEGREE) * rms * z, rms

    def forward(self, x):
        original = self.spectral(x)
        lifted, _ = self.lift(original, self.neighbors)
        return lifted


def attach(model, kind, graph):
    bank = model.body.spectral_filtering
    if (type(bank).__name__ != "FilterBankLayer" or bank.n_bands != BANDS or
            model.body.n_chans != SENSORS or model.body.n_times != SAMPLES or model.skip or
            model.dim != 1152 or model.task.in_features != 1439 or model.task.out_features != 3):
        raise ValueError("native FBCNet spectral or class head geometry differs")
    if kind == "hard":
        return None
    layer = ElectrodeLift(bank, graph, model.task.weight.device)
    model.body.spectral_filtering = layer
    return layer


def trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameters(kind):
    return BASE if kind == "hard" else BASE + EXTRA
