"""Waveform-to-label decoder. State is passed explicitly for each session."""
import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = nn.Conv1d(41, 48, 1, bias=False)
        self.temporal = nn.Conv1d(48, 48, 33, stride=4, groups=48, bias=False)
        self.write = nn.Sequential(nn.Linear(96, 48), nn.Tanh())
        self.read = nn.Sequential(nn.Linear(49, 64), nn.Tanh(), nn.Linear(64, 96))
        self.head = nn.Sequential(nn.Linear(192, 64), nn.GELU(), nn.Linear(64, 3))
        self.register_buffer('mean', torch.zeros(48))
        self.register_buffer('second', torch.ones(48))

    def encode(self, x):
        return self.temporal(self.spatial(x))

    def unpack(self, features):
        mu = features.mean(-1)
        power = features.square().mean(-1)
        return mu, power, self.write(torch.cat([mu, power.clamp_min(1e-8).log()], -1))

    def sequence(self, features, arm, initial=None, donor=None):
        mu, power, code = self.unpack(features)
        if self.training:
            with torch.no_grad():
                self.mean.lerp_(mu.mean(0), .04)
                self.second.lerp_(power.mean(0), .04)
        values = torch.cat([mu, power, code], -1)
        updates = values if donor is None else donor
        previous, count = (torch.zeros_like(values[:1]), 0) if initial is None else initial
        prefix = previous + torch.cat([torch.zeros_like(updates[:1]), updates[:-1].cumsum(0)])
        age = torch.arange(len(mu), device=mu.device, dtype=mu.dtype)[:, None] + count
        mean, energy, memory = prefix.split(48, -1)
        center = self.mean.detach().clone()
        second = self.second.detach().clone()
        if arm == 'norm':
            center = (20*center + mean)/(20+age)
            second = (20*second + energy)/(20+age)
        scale = (second-center.square()).clamp_min(1e-6).rsqrt()
        if arm != 'memory':
            memory = torch.zeros_like(memory)
            age = torch.zeros_like(age)
        context = self.read(torch.cat([memory/age.clamp_min(1), age/(age+20)], -1))
        bias, gain = context.chunk(2, -1)
        normalized = (features-center[...,None])*scale[...,None]
        adapted = normalized*gain.tanh().exp()[...,None] + bias[...,None]
        bins = adapted.unflatten(-1,(4,-1)).square().mean(-1).clamp_min(1e-8).log()
        return self.head(bins.flatten(1)), (previous+updates.sum(0,keepdim=True), count+len(mu))

    def step(self, x, arm, state=None):
        return self.sequence(self.encode(x), arm, state)
