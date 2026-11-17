"""Matched current-only and strictly previous-set residual decoders."""
import torch
from torch import nn


class Readout(nn.Module):
    def __init__(self):
        super().__init__()
        self.item = nn.Sequential(nn.Linear(1152, 16), nn.Tanh())
        self.head = nn.Sequential(nn.Linear(19, 16), nn.Tanh(), nn.Linear(16, 3))

    def correct(self, logp, context):
        return logp + .25 * self.head(torch.cat((logp, context), dim=-1)).tanh()

    def forward(self, logp, items, prefix):
        encoded = self.item(items)
        if prefix:
            previous = encoded[:, :-1].cumsum(dim=1)
            counts = torch.arange(1, encoded.shape[1], dtype=encoded.dtype)[None, :, None]
            encoded = torch.cat((torch.zeros_like(encoded[:, :1]), previous / counts), dim=1)
        return self.correct(logp, encoded)

    @torch.no_grad()
    def replay(self, probabilities, items, prefix, active=True):
        if not active:
            return probabilities.clone()
        state = Prefix(self) if prefix else None
        return torch.stack([state.step(p, x) if state else self.correct(p.log(), self.item(x)).softmax(-1)
                            for p, x in zip(probabilities, items)])


class Prefix:
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.total = torch.zeros(16, dtype=torch.float64)
        self.count = 0

    @torch.no_grad()
    def step(self, probability, item):
        result = self.model.correct(probability.log(), self.total / max(self.count, 1)).softmax(-1)
        self.total.add_(self.model.item(item))
        self.count += 1
        return result
