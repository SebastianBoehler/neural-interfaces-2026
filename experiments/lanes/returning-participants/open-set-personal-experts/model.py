"""Train the population prior and each single-visit EEG expert, then run causal inference."""
import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from features import VIEW_SIZES

SEED = 20261120


@dataclass
class Moments:
    mean: np.ndarray
    var: np.ndarray

    @classmethod
    def fit(cls, x):
        return cls(x.mean(0, dtype=np.float64), x.var(0, dtype=np.float64))

    def standardized(self, x):
        return np.clip((x - self.mean) / np.sqrt(np.maximum(self.var, 1e-8)), -5, 5).astype(np.float32)

    def prefix(self, x):
        # Shift before summing squares for numerical stability. Exactly eight prior windows.
        delta = np.asarray(x, np.float64) - self.mean
        n = 8 + np.arange(1, len(x) + 1, dtype=np.float64)[:, None]
        drift = delta.cumsum(0) / n
        var = (8 * self.var + np.square(delta).cumsum(0)) / n - drift**2
        return np.clip((delta - drift) / np.sqrt(np.maximum(var, 1e-8)), -5, 5).astype(np.float32)


def tensor(x):
    return torch.as_tensor(x, device="cuda", dtype=torch.float32)


def weights(y):
    count = torch.bincount(y, minlength=3)
    if (count == 0).any():
        raise ValueError("missing enrollment class")
    return len(y) / (3 * count[y])


def kernel(x, z):
    out = 0
    for a, b in zip(x.split(VIEW_SIZES, 1), z.split(VIEW_SIZES, 1)):
        cos = F.normalize(a) @ F.normalize(b).T
        out = out + .7 * cos + .3 * torch.exp(2 * (cos - 1))
    return out / len(VIEW_SIZES)


class Ridge:
    def __init__(self, x, y, targets, penalty):
        self.x = x
        gram = kernel(x, x).double()
        gram.diagonal().add_(penalty / weights(y))
        self.alpha = torch.linalg.solve(gram, targets.double()).float()

    def __call__(self, x):
        return kernel(x, self.x) @ self.alpha


def train(net, x, y, epochs, lr):
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=.01)
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    balance = weights(y)
    net.train()
    for _ in range(epochs):
        for rows in torch.randperm(len(y), device="cuda", generator=gen).split(64):
            opt.zero_grad()
            loss = (F.cross_entropy(net(x[rows]), y[rows], reduction="none") * balance[rows]).mean()
            loss.backward()
            opt.step()
    return net.eval()


def enrollment(d):
    x = d["f"][d["day"] == 1]
    y = torch.as_tensor(d["y1"], dtype=torch.long, device="cuda")
    assert len(x) == len(y)
    m = Moments.fit(x)
    return m, tensor(m.standardized(x)), y


class Population:
    def __init__(self, people):
        pairs = [enrollment(d)[1:] for d in people]
        x = torch.cat([a for a, _ in pairs])
        y = torch.cat([b for _, b in pairs])
        torch.manual_seed(SEED)
        self.ridge = Ridge(x, y, F.one_hot(y, 3).float(), .25)
        self.net = train(nn.Sequential(nn.Linear(x.shape[1], 128), nn.LayerNorm(128), nn.GELU(),
                                       nn.Dropout(.35), nn.Linear(128, 3)).cuda(), x, y, 32, 3e-4)

    def personalize(self, person):
        return Expert(self, person)


class Expert:
    def __init__(self, pop, person):
        self.m, x, y = enrollment(person)
        self.pop = pop
        # A personal residual can correct a population class map already supported by the individual.
        with torch.no_grad():
            residual = F.one_hot(y, 3).float() - .35 * pop.ridge(x)
        self.ridge = Ridge(x, y, residual, .12)
        torch.manual_seed(SEED)
        self.net = train(copy.deepcopy(pop.net), x, y, 27, 2e-4)

    @torch.no_grad()
    def predict(self, f):
        x = tensor(self.m.prefix(f))
        ridge = (.35 * self.pop.ridge(x) + self.ridge(x)) * 5
        deep = self.net(x)
        return torch.stack([ridge.softmax(-1), deep.softmax(-1)], 1).cpu().numpy()
