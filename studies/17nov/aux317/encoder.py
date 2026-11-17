"""Independent task encoders with an identical information-preserving readout."""
import hashlib

import numpy as np
import torch
from torch import nn

BANDS = ((.1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 55))


def moments(x):
    f = torch.fft.rfftfreq(480, 1 / 120).to(x.device)
    p = torch.fft.rfft(x).abs().square() / 480 ** 2
    return torch.cat([x.mean(-1)] + [p[:, :, (f >= lo) & (f < hi)].mean(-1)
                     .clamp_min(1e-12).log() for lo, hi in BANDS], 1)


def fingerprint(net):
    h = hashlib.sha256()
    for name, value in sorted(net.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class Encoder(nn.Module):
    def __init__(self, arch, seed, device):
        super().__init__()
        import braindecode.models as bd
        torch.manual_seed(seed)
        cls = {'fb': bd.FBCNet, 'sh': bd.ShallowFBCSPNet}[arch]
        self.body = cls(n_chans=41, n_outputs=3, n_times=480, sfreq=120)
        self.body.final_layer = nn.Identity()
        self.frozen = False
        self.to(device).eval()
        with torch.no_grad():
            self.dim = self.body(torch.ones(2, 41, 480, device=device)).flatten(1).shape[1]
        self.task = nn.Linear(self.dim + 287, 3).to(device)
        self.reconstruct = nn.Linear(self.dim, 287).to(device)
        self.register_buffer('moment_mean', torch.zeros(287, device=device))
        self.register_buffer('moment_sd', torch.ones(287, device=device))
        self.frozen = False

    def representation(self, x):
        return self.body(x).flatten(1)

    def forward(self, x):
        z = self.representation(x)
        skip = (moments(x) - self.moment_mean) / self.moment_sd
        return self.task(torch.cat([z, skip], 1))

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.body.eval()
        return self

