"""Bounded identity-initialized spatial/filter/contrast display,295 parameters."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from support325 import Display, digest


class Renderer(Display):
    def __init__(self, base, neighbors):
        super().__init__(float(base.scale), base.projection.cpu().numpy())
        self.register_buffer('neighbors', torch.as_tensor(neighbors, dtype=torch.long))
        self.reference = nn.Parameter(torch.zeros(41))
        self.spatial = nn.Parameter(torch.zeros(41, 4))
        self.filter_gain = nn.Parameter(torch.zeros(8))
        self.contrast = nn.Parameter(torch.zeros(41))
        self.offset = nn.Parameter(torch.zeros(41))
        knots = torch.logspace(np.log10(.5), np.log10(55), 8)
        freq = torch.fft.rfftfreq(2048, 1 / 120)
        position = ((freq.clamp_min(.5).log() - knots[0].log()) /
                    (knots[-1].log() - knots[0].log()) * 7).clamp(0, 7)
        low = position.floor().long().clamp_max(6)
        self.register_buffer('low', low)
        self.register_buffer('fraction', position - low)
        if sum(p.numel() for p in self.parameters()) != 295: raise ValueError('renderer size changed')

    def transformed(self, x):
        if x.shape[1:] != (41, 480): raise ValueError('original41x480required')
        v = x / self.scale
        v = v - .25 * self.reference.tanh()[None, :, None] * v.mean(1, keepdim=True)
        v = v - (.25 * self.spatial.tanh()[None, :, :, None] * (x / self.scale)[:, self.neighbors]).sum(2)
        gain = np.log(2) * self.filter_gain.tanh()
        response = torch.exp(gain[self.low] * (1-self.fraction) + gain[self.low+1] * self.fraction)
        response = torch.cat([torch.ones_like(response[:1]), response[1:]])
        v = torch.fft.irfft(torch.fft.rfft(F.pad(v, (784,784))) * response, n=2048)[..., 784:1264]
        return v

    def intensity(self, x):
        v = self.transformed(x) * torch.exp(np.log(2) * self.contrast.tanh())[None, :, None]
        plane = .5 + torch.atan(v) / torch.pi + .1 * self.offset.tanh()[None, :, None]
        return plane.clamp(0, 1)

    def forward(self, x):
        plane = self.intensity(x).unsqueeze(2).expand(-1, -1, 14, -1).reshape(len(x), 1, 574, 480)
        image = F.pad(plane, (0,10,0,0), value=.5).repeat(1,3,1,1)
        return (image-self.rgb_mean)/self.rgb_sd

    def direct(self, x):
        return self.intensity(x).reshape(len(x),41,48,10).mean(-1).flatten(1)


def neighbors(names):
    import mne
    positions = mne.channels.make_standard_montage('standard_1005').get_positions()['ch_pos']
    xyz = np.stack([positions[str(n)] for n in names])
    distance = ((xyz[:,None]-xyz[None])**2).sum(2)
    np.fill_diagonal(distance, np.inf)
    index = np.argsort(distance, axis=1, kind='stable')[:, :4]
    return index, dict(mne_version=mne.__version__, coordinates_sha256=digest(xyz),
                       neighbors_sha256=digest(index), names=[str(n) for n in names])
