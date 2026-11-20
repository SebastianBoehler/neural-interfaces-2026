"""Build causal embedding and similarity banks from shuffled sessions."""
from dataclasses import dataclass

import numpy as np
import torch

from base import C
from context import MAX_PREFIX, capture

DEVICE = "cuda:0"


@torch.no_grad()
def aligned(x, shrink=C.DEFAULT_TTT["shrink"]):
    x = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
    x64 = x.double()
    cov = (x64 @ x64.transpose(1, 2)) / x.shape[-1]
    age = torch.arange(1, len(x) + 1, device=DEVICE)
    return C.inv_sqrt(cov.cumsum(0) / age[:, None, None], shrink) @ x


@dataclass
class Episode:
    logits: torch.Tensor
    similarity: torch.Tensor
    labels: np.ndarray


@torch.no_grad()
def encode(base, x, labels):
    x = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
    vectors, logits = [], []
    for small in x.split(24):
        _, z, scores = capture(base.nets, lambda: [net(small) for net in base.nets])
        vectors.append(z)
        logits.append(scores)
    members = [torch.cat(parts) for parts in zip(*vectors)]
    sim = torch.stack([z @ z.T for z in members], dim=-1)
    return Episode(torch.cat(logits).half().cpu(), sim.half().cpu(), labels)


def target_bank(base, x, y, groups, rng, orders):
    episodes = []
    for group in sorted(set(groups)):
        rows = np.flatnonzero(groups == group)
        for _ in range(orders):
            p = rng.permutation(rows)[:MAX_PREFIX + 1]
            episodes.append(encode(base, aligned(x[p]), y[p]))
    if not episodes:
        raise ValueError("no target episodes")
    return episodes
