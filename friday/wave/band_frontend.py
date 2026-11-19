"""Matched random or EEG band-pass initialization for a learned raw front end."""
import hashlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from normalize import NormalizedRawDecoder
from raw import TOKEN_WIDTH
from waveform import SENSORS


KINDS = ("raw_band_spectral", "raw_band_random")
BANDS = ((.1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 55))
TAPS = 129
SAMPLE_RATE = 120


class PowerStem(nn.Module):
    def __init__(self, spectral):
        super().__init__()
        bands = len(BANDS)
        self.spatial = nn.Conv1d(SENSORS, TOKEN_WIDTH, 1, bias=False)
        self.temporal = nn.Conv1d(bands, bands, TAPS, padding=TAPS // 2,
                                  groups=bands, bias=False)
        if spectral:
            self.initialize_bandpasses()

    @torch.no_grad()
    def initialize_bandpasses(self):
        filt = self.temporal.weight
        offsets = torch.arange(TAPS, dtype=torch.float64) - TAPS // 2
        taper = torch.hann_window(TAPS, periodic=False, dtype=offsets.dtype)
        edges = torch.tensor(BANDS, dtype=offsets.dtype) * (2 / SAMPLE_RATE)
        pulses = edges[..., None] * torch.sinc(edges[..., None] * offsets)
        bandpass = (pulses[:, 1] - pulses[:, 0]) * taper
        bandpass = F.normalize(bandpass, dim=1).to(filt)
        filt.copy_(bandpass[:, None] * filt.norm(dim=-1, keepdim=True))

    def forward(self, eeg):
        batch, _, samples = eeg.shape
        bands = len(BANDS)
        copies = TOKEN_WIDTH // bands
        # Shared temporal filters commute with the bias-free in-band spatial projections.
        projected = self.spatial(eeg).reshape(batch, bands, copies, samples)
        projected = projected.transpose(1, 2).reshape(batch * copies, bands, samples)
        filtered = self.temporal(projected).reshape(batch, copies, bands, samples)
        signal = filtered.transpose(1, 2).reshape(batch, TOKEN_WIDTH, samples)
        return F.avg_pool1d(signal.square(), 6).clamp_min(1e-8).sqrt()


class BandRawDecoder(NormalizedRawDecoder):
    def __init__(self, kind):
        if kind not in KINDS:
            raise ValueError(f"unknown filter-bank initialization {kind}")
        super().__init__("raw_norm_fixed")
        self.stem = PowerStem(spectral=kind == KINDS[0])


def tensor_bytes(tensor):
    return np.asarray(tensor.detach().cpu().numpy(), dtype="<f4").tobytes()


def fir_hash(model):
    return hashlib.sha256(tensor_bytes(model.stem.temporal.weight)).hexdigest()


def initial_receipt(model, kind):
    other = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name != "stem.temporal.weight":
            other.update(name.encode() + b"\0")
            other.update(tensor_bytes(value))
    weight = model.stem.temporal.weight.detach().cpu().double()[:, 0]
    power = torch.fft.rfft(weight, n=4096).abs().square()
    frequencies = torch.fft.rfftfreq(4096, 1 / SAMPLE_RATE)
    in_band = [float(p[(frequencies >= lo) & (frequencies < hi)].sum() / p.sum())
               for p, (lo, hi) in zip(power, BANDS, strict=True)]
    if kind == KINDS[0] and min(in_band) <= .75:
        raise ValueError(f"initial bandpass concentration below 0.75: {in_band}")
    if kind == KINDS[1] and max(in_band) >= .55:
        raise ValueError(f"random control unexpectedly band-concentrated: {in_band}")
    return dict(initial_fir=fir_hash(model), initial_other=other.hexdigest(),
                fir_norm=weight.norm(dim=-1).tolist(), in_band=in_band,
                parameters=sum(p.numel() for p in model.parameters()))
