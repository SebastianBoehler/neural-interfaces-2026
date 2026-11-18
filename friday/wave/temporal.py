"""Compare signed and time-reversal-invariant within-window spectra."""

import numpy as np
import torch
from torch import nn

from waveform import BANDS, PROJECTIONS, SENSORS, WIDTH, Decoder


KINDS = ("chronological", "symmetric", "coherent_time", "scattered_time")


class TemporalDecoder(Decoder):
    def __init__(self, kind):
        if kind not in KINDS:
            raise ValueError(f"unknown temporal decoder {kind}")
        super().__init__("prototype")
        self.symmetric = kind == "symmetric"
        self.register_buffer("frame_taper", torch.hann_window(120))
        edges = [edge - 1 for edge in BANDS]
        weights = torch.zeros(edges[-1], len(edges) - 1)
        for i, (start, stop) in enumerate(zip(edges, edges[1:])):
            weights[start:stop, i] = 1 / (stop - start)
        self.register_buffer("frame_weights", weights)
        size = 4 * (SENSORS + PROJECTIONS) * (len(BANDS) - 1)
        self.encoder = nn.Sequential(nn.LayerNorm(size), nn.Linear(size, 64),
                                     nn.GELU(), nn.Linear(64, WIDTH), nn.LayerNorm(WIDTH))

    def spectrum(self, x):
        full = super().spectrum(x)
        frames = x.reshape(len(x), SENSORS, 4, 120)
        frames = torch.fft.rfft(frames * self.frame_taper)[..., 1:45]
        return torch.cat([full, frames.flatten(2)], dim=-1)

    def time_features(self, frames):
        late = (frames[..., 2, :] + frames[..., 3, :]) / 2
        early = (frames[..., 0, :] + frames[..., 1, :]) / 2
        direction = late - early
        symmetric = (frames[..., 1, :] + frames[..., 2, :]) / 2
        symmetric = symmetric - (frames[..., 0, :] + frames[..., 3, :]) / 2
        return torch.cat([direction.abs() if self.symmetric else direction, symmetric], dim=-1)

    def encode(self, spectrum):
        power = self.project_power(spectrum)
        full, frames = power.split((self.band_weights.shape[0], 4 * self.frame_weights.shape[0]), dim=-1)
        bands = (full @ self.band_weights).clamp_min(1e-6).log()
        relative = bands - bands.mean(dim=-1, keepdim=True)
        frame_bands = (frames.reshape(*frames.shape[:2], 4, -1) @ self.frame_weights).clamp_min(1e-6).log()
        time = self.time_features(frame_bands)
        return self.encoder(torch.cat([bands, relative, time], dim=-1).flatten(1))


class SourceJitter:
    def __init__(self, kind, seed, steps):
        self.coherent = kind == "coherent_time"
        rng = np.random.default_rng(seed + 10000)
        self.schedule = np.stack([rng.permutation(np.array([-30, -15, 0, 15, 30]))
                                  for _ in range((steps + 4) // 5)])
        if kind not in ("coherent_time", "scattered_time"):
            raise ValueError(f"unknown source jitter {kind}")

    def __call__(self, source, update):
        positions = np.zeros(len(source), dtype=int) if self.coherent else np.arange(len(source))
        offsets = self.schedule[update // 5, (update + positions) % 5]
        shifts = torch.from_numpy(offsets).to(source.device)
        time = (torch.arange(480, device=source.device) + shifts[:, None]).abs()
        time = torch.minimum(time, 2 * 479 - time)
        return torch.gather(source, -1, time[:, None].expand(-1, source.size(1), -1))
