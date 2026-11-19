"""Lane C4: frozen foundation-model encoder + small trainable head, in Lane C's member interface.

The body is frozen (no gradient, always eval, forward under no_grad); only the token BatchNorm, a 128-wide projection, the electrode-position
encoder and an eight-query attention-pooling head (score, task, motor) train. Optimizer groups keep Lane C's two-group shape (BatchNorm 1e-4, heads 1e-3)
so its cosine and OneCycle schedules apply unchanged. Weight decay 0.05 is the regulariser.
"""
import torch
from torch import nn

from oracle import require


class HeadMember(nn.Module):
    """Frozen body + channel-aware head (form 2). Form 1 (LayerNorm + one-query attention pooling, no position input) could not fit
    even the training rows (loss about ln 3, held BA 35-38 %): a pool over channels discards the lateralisation a frozen encoder leaves in
    per-channel tokens. Form 2 adds each token's electrode position (a small encoder) before eight-query attention pooling.
    Subclasses give `tokens(x, pos) -> (tokens [B, N, dim], where [N, ...])` and the position encoder."""

    def __init__(self, body, dim, pos_encoder, width=128, queries=8):
        super().__init__()
        self.body = body
        for p in self.body.parameters():
            p.requires_grad_(False)
        self.norm = nn.BatchNorm1d(dim)  # feature-wise standardisation of frozen tokens (form 3; form 2's per-token LayerNorm left NeuroRVQ at loss ln 3)
        self.proj = nn.Linear(dim, width)
        self.pos = pos_encoder
        self.score = nn.Linear(width, queries)
        self.drop = nn.Dropout(0.1)
        self.task = nn.Linear(queries * width, 4)
        self.motor = nn.Linear(queries * width, 5)

    def train(self, mode=True):
        super().train(mode)
        self.body.eval()  # frozen body: no dropout, fixed statistics
        return self

    def tokens(self, x, pos):  # [B, C, T] at 120 Hz, positions -> ([B, N, dim], per-token position input [N, ...])
        raise NotImplementedError

    def forward(self, x, pos):
        with torch.no_grad():
            tokens, where = self.tokens(x, pos)
        h = torch.nn.functional.gelu(self.proj(self.norm(tokens.float().transpose(1, 2)).transpose(1, 2)) + self.pos(where))
        pooled = self.drop(torch.einsum("bnq,bnd->bqd", self.score(h).softmax(1), h).flatten(1))
        return self.task(pooled), self.motor(pooled)


def split_optimizer(model):
    require(not any(p.requires_grad for p in model.body.parameters()), "encoder is not frozen")
    first = list(model.norm.parameters())
    rest = [p for m in (model.proj, model.pos, model.score, model.task, model.motor) for p in m.parameters()]
    require(len(first) + len(rest) == sum(p.requires_grad for p in model.parameters()), "optimizer groups do not partition")
    return torch.optim.AdamW([{"params": first, "lr": 1e-4}, {"params": rest, "lr": 1e-3}], weight_decay=0.05)


def zscore(x):
    """Per channel and window, like the codec and Pratibha inputs (correction=1)."""
    return (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True, correction=1) + 1e-6)
