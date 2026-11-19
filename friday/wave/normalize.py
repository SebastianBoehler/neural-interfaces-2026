"""Causal embedding normalization with a trained or fixed source prior."""

import torch
from torch import nn

from raw import RawDecoder
from waveform import WIDTH


KINDS = ("raw_norm_fixed", "raw_norm_learned")


class NormalizedRawDecoder(RawDecoder):
    def __init__(self, kind):
        if kind not in KINDS:
            raise ValueError(f"unknown normalization mode {kind}")
        super().__init__()
        log_count = torch.zeros(WIDTH)
        if kind == "raw_norm_learned":
            self.log_count = nn.Parameter(log_count)
        else:
            self.register_buffer("log_count", log_count)

    def prior(self):
        return 20 * self.log_count.clamp(-2, 2).exp()

    def writes(self, z, logits):
        return torch.cat([super().writes(z, logits), z.square()], dim=-1)

    def read(self, z, logits, history, age):
        previous, second = history[:, :-WIDTH], history[:, -WIDTH:]
        result = super().read(z, logits, previous, age)
        first = previous[:, -WIDTH:]
        count = self.prior()
        mass = age[:, None] + count
        mean = first / mass
        variance = (second + count) / mass - mean.square()
        corrected = (z - mean) * variance.clamp_min(.25).rsqrt()
        delta = (corrected - z).clamp(-2, 2)
        return result + delta @ self.static.weight.T
