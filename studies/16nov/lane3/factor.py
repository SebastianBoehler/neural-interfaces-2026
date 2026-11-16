"""Small independent task/style experiment; identity is training-only."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def person_batches(rows, people, streams, rng, chunk=16):
    """Cover each row, pairing single-session chunks from different people."""
    rows = np.asarray(rows)
    people, streams = np.asarray(people), np.asarray(streams)
    if chunk < 2 or len(np.unique(people[rows])) < 2:
        raise ValueError("two distinct people and at least two rows per chunk required")
    groups = {}
    for person in np.unique(people[rows]):
        selected = rows[people[rows] == person]
        chunks = []
        for stream in np.unique(streams[selected]):
            indices = selected[streams[selected] == stream].copy()
            rng.shuffle(indices)
            for start in range(0, len(indices), chunk):
                block = indices[start:start + chunk]
                if len(block) < chunk:
                    block = np.concatenate((block, rng.choice(indices, chunk - len(block), replace=True)))
                chunks.append(block)
        rng.shuffle(chunks)
        groups[person] = chunks
    original = {p: tuple(v) for p, v in groups.items()}
    result = []
    while any(groups.values()):
        candidates = list(groups)
        rng.shuffle(candidates)
        candidates.sort(key=lambda p: len(groups[p]), reverse=True)
        first, second = candidates[:2]
        left = groups[first].pop() if groups[first] else original[first][rng.integers(len(original[first]))]
        right = groups[second].pop() if groups[second] else original[second][rng.integers(len(original[second]))]
        result.append(np.concatenate((left, right)))
    return result


def contrastive(style, people, temperature=0.2):
    """Supervised contrast: every anchor needs same- and different-person examples."""
    if style.ndim != 2 or len(style) != len(people) or temperature <= 0:
        raise ValueError("invalid style, people, or temperature")
    eye = torch.eye(len(style), device=style.device, dtype=torch.bool)
    same = people[:, None].eq(people[None, :]) & ~eye
    other = ~people[:, None].eq(people[None, :])
    if not bool(same.any(1).all() and other.any(1).all()):
        raise ValueError("each anchor needs another same-person and a different-person window")
    z = F.normalize(style, dim=-1)
    scores = (z @ z.T / temperature).masked_fill(eye, -torch.inf)
    log_prob = scores - torch.logsumexp(scores, dim=1, keepdim=True)
    return -(log_prob.masked_fill(~same, 0).sum(1) / same.sum(1)).mean()


class FactorNet(nn.Module):
    """Use the original CBraMod head; independent tokenwise residual branches."""
    def __init__(self, backbone, arm, dim=200, width=32):
        super().__init__()
        if arm not in ("off", "matched", "split"):
            raise ValueError(f"unknown arm: {arm}")
        if backbone.fwd:
            raise ValueError("unexpected backbone forward options")
        self.arm = arm
        self.head = backbone.net.final_layer
        backbone.net.final_layer = nn.Identity()
        backbone.fwd = {"return_features": True}
        self.backbone = backbone
        if arm != "off":
            self.task = nn.Sequential(nn.Linear(dim, width), nn.GELU(), nn.Linear(width, dim))
            self.style = nn.Sequential(nn.Linear(dim, width), nn.GELU(), nn.Linear(width, dim))

    def forward(self, x, people=None):
        tokens = self.backbone(x)["features"]
        if self.arm == "off":
            return self.head(tokens), tokens.new_zeros(())
        task = tokens + self.task(tokens)
        style = self.style(tokens)
        if self.arm == "matched":
            return self.head(task + style), tokens.new_zeros(())
        auxiliary = tokens.new_zeros(())
        if people is not None:
            pooled = style.mean(dim=tuple(range(1, style.ndim - 1)))
            auxiliary = 0.05 * (contrastive(pooled, people) + F.mse_loss(task + style, tokens.detach()))
        return self.head(task), auxiliary
