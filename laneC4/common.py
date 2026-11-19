"""Lane C4: frozen foundation-model encoder + small trainable head, in Lane C's member interface.

The body is frozen (no gradient, always eval, forward under no_grad); only a LayerNorm over tokens and the learned-query
attention-pooling head (score, task, motor) train. Optimizer groups keep Lane C's two-group shape (LayerNorm 1e-4, heads 1e-3)
so its cosine and OneCycle schedules apply unchanged. Weight decay 0.05 is the regulariser.
"""
import torch
from torch import nn

from oracle import require


class HeadMember(nn.Module):
    def __init__(self, body, dim):
        super().__init__()
        self.body = body
        for p in self.body.parameters():
            p.requires_grad_(False)
        self.norm = nn.LayerNorm(dim)
        self.score = nn.Linear(dim, 1)
        self.task = nn.Linear(dim, 4)
        self.motor = nn.Linear(dim, 5)

    def train(self, mode=True):
        super().train(mode)
        self.body.eval()  # frozen body: no dropout, fixed statistics
        return self

    def tokens(self, x, pos):  # [B, C, T] at 120 Hz, positions -> [B, N, dim]
        raise NotImplementedError

    def forward(self, x, pos):
        with torch.no_grad():
            tokens = self.tokens(x, pos)
        tokens = self.norm(tokens.float())
        pooled = (self.score(tokens).softmax(1) * tokens).sum(1)
        return self.task(pooled), self.motor(pooled)


def split_optimizer(model):
    require(not any(p.requires_grad for p in model.body.parameters()), "encoder is not frozen")
    first = list(model.norm.parameters())
    rest = list(model.score.parameters()) + list(model.task.parameters()) + list(model.motor.parameters())
    require(len(first) + len(rest) == sum(p.requires_grad for p in model.parameters()), "optimizer groups do not partition")
    return torch.optim.AdamW([{"params": first, "lr": 1e-4}, {"params": rest, "lr": 1e-3}], weight_decay=0.05)


def zscore(x):
    """Per channel and window, like the codec and Pratibha inputs (correction=1)."""
    return (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True, correction=1) + 1e-6)
