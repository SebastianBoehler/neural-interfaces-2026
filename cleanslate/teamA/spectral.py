"""Spectral input tokens for the full-depth MANAS-2 variant."""
import torch
from torch import nn
from torch.nn import functional as F


class SpectralTokens(nn.Module):
    def __init__(self, width=512):
        super().__init__()
        self.proj = nn.Linear(14, width)
        nn.init.normal_(self.proj.weight, std=0.006)
        nn.init.zeros_(self.proj.bias)
        self.register_buffer("taper", torch.hann_window(200))

    def forward(self, x):
        patches = x.unfold(-1, 200, 180).float()
        # Raw-channel and common-average-referenced spectra retain complementary spatial structure.
        referenced = patches - patches.mean(1, keepdim=True)
        spectrum = torch.fft.rfft(torch.stack((patches, referenced), -2) * self.taper, dim=-1)
        power = spectrum.real.square() + spectrum.imag.square()
        bands = tuple(power[..., lo:hi].mean(-1) for lo, hi in
                      ((1, 4), (4, 8), (8, 13), (13, 20), (20, 32), (32, 45), (45, 59)))
        energy = torch.stack(bands, -1).clamp_min(1e-5).log()
        return self.proj((energy.flatten(-2) / 4).clamp(-4, 4))


class SpectralDecoder(Decoder):
    def __init__(self, arch, names, positions=None):
        if arch != "manas":
            raise ValueError("The spectral token form uses MANAS-2")
        super().__init__(arch, names, positions)
        self.spectral = SpectralTokens()

    def features(self, x):
        wave = F.interpolate(x, size=800, mode="linear", align_corners=False).clamp(-15, 15)
        tokens = self.encoder(wave, self.positions.expand(len(wave), -1, -1),
                              self.spectral(wave)).transpose(1, 2)
        features = self.token(tokens)
        mean = features.mean(2)
        variability = (features.var(2, unbiased=False) + 1e-4).sqrt()
        return self.spatial(torch.cat((mean, variability), -1))
