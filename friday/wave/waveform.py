"""Trainable frequency-space encoder and fixed-size causal session state."""

import torch
from torch import nn
from torch.nn import functional as F


BANDS = (1, 4, 7, 10, 13, 18, 25, 35, 45)
SENSORS = 41
PROJECTIONS = 16
WIDTH = 32


def center(logits):
    return logits - logits.mean(dim=-1, keepdim=True)


class SessionDecoder(nn.Module):
    def init_head(self):
        self.static = nn.Linear(WIDTH, 3)
        self.templates = nn.Parameter(torch.zeros(3, WIDTH))
        self.style = nn.Linear(WIDTH, 3, bias=False)
        self.prototype_scale = nn.Parameter(torch.ones(3))

    def writes(self, z, logits):
        p = logits.detach().softmax(dim=-1)
        confidence = p.square()
        prototypes = confidence.unsqueeze(-1) * z.unsqueeze(1)
        style = z - p @ self.templates.detach() if self.centered else z
        return torch.cat([confidence, prototypes.flatten(1), style], dim=-1)

    def read(self, z, logits, history, age):
        mass, prototypes, style = history.split((3, 3 * WIDTH, WIDTH), dim=-1)
        prototypes = prototypes.reshape(-1, 3, WIDTH) / (mass.unsqueeze(-1) + 3)
        prototypes = prototypes - prototypes.mean(dim=1, keepdim=True)
        class_match = (z.unsqueeze(1) * prototypes).mean(dim=-1)
        style_match = self.style(z * style / (age.unsqueeze(-1) + 12))
        return logits + self.prototype_scale * class_match + style_match

    def sequence(self, z, logits, writes):
        history = torch.cat([torch.zeros_like(writes[:1]), writes[:-1].cumsum(dim=0)])
        age = torch.arange(len(z), device=z.device)
        return self.read(z, logits, history, age)

    def step(self, x, state=None):
        z = self.encode(self.prepare(x))
        logits = self.static(z)
        writes = self.writes(z, logits)
        history, age = state if state is not None else (torch.zeros_like(writes), 0)
        result = self.read(z, logits, history, logits.new_tensor([age]))
        return result, (history + writes, age + 1)


class Decoder(SessionDecoder):
    def __init__(self, kind):
        super().__init__()
        if kind not in ("prototype", "centered"):
            raise ValueError(f"unknown decoder {kind}")
        self.centered = kind == "centered"
        self.spatial = nn.Parameter(torch.randn(PROJECTIONS, SENSORS) / SENSORS**.5)
        self.register_buffer("taper", torch.hann_window(480))
        edges = [4 * edge - 4 for edge in BANDS]
        weights = torch.zeros(edges[-1], len(edges) - 1)
        for i, (start, stop) in enumerate(zip(edges, edges[1:])):
            weights[start:stop, i] = 1 / (stop - start)
        self.register_buffer("band_weights", weights)
        size = 2 * (SENSORS + PROJECTIONS) * (len(BANDS) - 1)
        self.encoder = nn.Sequential(nn.LayerNorm(size), nn.Linear(size, 64),
                                     nn.GELU(), nn.Linear(64, WIDTH), nn.LayerNorm(WIDTH))
        self.init_head()

    def prepare(self, x):
        return self.spectrum(x)

    def spectrum(self, x):
        return torch.fft.rfft(x * self.taper)[..., 4:180].contiguous()

    def project_power(self, spectrum):
        spatial = F.normalize(self.spatial, dim=-1).to(spectrum.dtype)
        signals = torch.cat((spectrum, spatial @ spectrum), dim=1)
        return signals.real.square() + signals.imag.square()

    def encode(self, spectrum):
        bands = (self.project_power(spectrum) @ self.band_weights).clamp_min(1e-6).log()
        relative = bands - bands.mean(dim=-1, keepdim=True)
        return self.encoder(torch.cat([bands, relative], dim=-1).flatten(1))
