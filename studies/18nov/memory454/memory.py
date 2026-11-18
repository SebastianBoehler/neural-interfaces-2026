"""Causal session memory. The caller owns the prefix sum and resets each session."""
import torch
from torch import nn
from torch.nn import functional as F


class SpectralMemory(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.encoder = nn.Sequential(nn.Linear(246, 64), nn.LayerNorm(64), nn.GELU())
        self.head = nn.Linear(64, 3)
        self.query = nn.Linear(64, 12)
        self.key = nn.Linear(64, 12)
        self.value = nn.Linear(64, 3) if kind == 'kernel' else nn.Linear(12, 3, bias=False)

    def writes(self, encoded):
        key = self.key(encoded)
        if self.kind == 'kernel':
            key = F.elu(key) + 1
            products = key[:, :, None] * self.value(encoded)[:, None, :]
            return torch.cat((products.flatten(1), key), dim=1)
        key = key.tanh()
        return (key[:, :, None] * key[:, None, :]).flatten(1)

    def sequence(self, encoded, writes, state=None):
        previous, age = (torch.zeros_like(writes[:1]), 0) if state is None else state
        past = previous + torch.cat((torch.zeros_like(writes[:1]), writes[:-1].cumsum(0)))
        query = self.query(encoded)
        if self.kind == 'kernel':
            query = (F.elu(query) + 1) / 12
            table, mass = past.split((36, 12), dim=1)
            correction = (query[:, :, None] * table.reshape(-1, 12, 3)).sum(1)
            correction = correction / (20 + (query * mass).sum(1, keepdim=True))
        else:
            count = age + torch.arange(len(query), device=query.device)[:, None]
            table = past.reshape(-1, 12, 12)
            context = (query.tanh()[:, :, None] * table).sum(1) / (20 + count)
            correction = self.value(context)
        return self.head(encoded) + correction, (previous + writes.sum(0, keepdim=True), age + len(writes))
