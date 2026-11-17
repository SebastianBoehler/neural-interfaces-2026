"""Two spatial projections with aligned short/long FIR features and an explicit session prior."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from model import Decoder

ARMS = ('raw', 'spec', 'perm')
CHANNELS = 24


class DualScale(Decoder):
    def __init__(self, arm):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm = arm
        self.temporal = nn.Conv1d(CHANNELS, CHANNELS, 33, stride=4, groups=CHANNELS, bias=False)
        self.mixer = nn.Conv1d(2*CHANNELS, CHANNELS, 1)
        self.taps = nn.ParameterList()
        hz = torch.tensor([4., 6., 8., 10., 12., 16., 22., 32.]).repeat(3)[:,None]
        for size in (33, 97):
            self.taps.append(nn.Parameter(torch.randn(CHANNELS, 1, size) / math.sqrt(size)))
            phase = (2*torch.pi/120)*hz*(torch.arange(size)-(size-1)/2)
            window = torch.hann_window(size, periodic=False)
            self.register_buffer('real'+str(size), (phase.cos()*window)[:,None])
            self.register_buffer('imag'+str(size), (phase.sin()*window)[:,None])
        gen = torch.Generator().manual_seed(380)
        order = torch.cat([28*i+torch.randperm(28, generator=gen) for i in range(4)])
        self.register_buffer('permutation', order)

    def branch(self, signal, taps):
        size = taps.shape[-1]
        pad = (size-33)//2
        if self.arm == 'raw':
            return F.conv1d(signal, taps, stride=4, padding=pad, groups=CHANNELS).asinh()
        envelope = F.softplus(taps) / math.sqrt(size)
        kernel = torch.cat([envelope*getattr(self, 'real'+str(size)),
                            envelope*getattr(self, 'imag'+str(size))], dim=1)
        response = F.conv1d(signal, kernel.reshape(-1,1,size), stride=4,
                            padding=pad, groups=CHANNELS)
        energy = response.unflatten(1,(CHANNELS,2)).square().sum(2)
        return energy.clamp_min(1e-12).sqrt().log1p()

    def encode(self, x):
        direct, analysis = self.spatial(x).chunk(2, dim=1)
        short, long = [self.branch(analysis,t) for t in self.taps]
        if self.arm == 'perm':
            short = short.index_select(-1,self.permutation)
        blend = self.mixer(torch.cat([short,long],dim=1)).sigmoid()
        return torch.cat([self.temporal(direct), blend*short+(1-blend)*long],dim=1)

    def sequence(self, features, arm, initial=None, donor=None):
        if arm != self.arm or donor is not None:
            raise ValueError('unexpected arm or donor')
        return super().sequence(features,'norm',initial)
