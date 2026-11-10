"""Compact learned patch and cross-view models for source folds."""

import numpy as np
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from readouts import logits, shuffle_tiles


class PatchNet(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.project = nn.Conv1d(width, 24, 3, padding=1)
        self.temporal = nn.Conv1d(24, 24, 3, padding=1, groups=24)
        self.head = nn.Linear(24, 3)

    def forward(self, x):
        x = torch.nn.functional.gelu(self.project(x.transpose(1, 2)))
        x = x + torch.nn.functional.gelu(self.temporal(x))
        return self.head(x.mean(dim=-1))


class CrossView(nn.Module):
    def __init__(self, rank_width, time_width, product):
        super().__init__()
        self.rank = nn.Linear(rank_width, 24)
        self.temporal = nn.Linear(time_width, 24)
        self.product = product
        self.head = nn.Linear(72, 3)

    def forward(self, rank, time):
        a = torch.tanh(self.rank(rank))
        b = torch.tanh(self.temporal(time))
        interaction = a * b if self.product else .5 * (a + b)
        return self.head(torch.cat((a, b, interaction), dim=1))


def fit_predict(factory, training, held, labels, seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed + 9000)
    model = factory()
    optim = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.02)
    loss = nn.CrossEntropyLoss()
    t = tuple(torch.as_tensor(x, dtype=torch.float32) for x in training)
    v = tuple(torch.as_tensor(x, dtype=torch.float32) for x in held)
    y = torch.as_tensor(labels, dtype=torch.long)
    for _ in range(24):
        for rows in torch.randperm(len(y), generator=rng).split(96):
            optim.zero_grad()
            loss(model(*(x[rows] for x in t)), y[rows]).backward()
            optim.step()
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(*v), dim=1).numpy().astype(np.float64)


def patch(bank, draw, held, y, people, interface, seed, fold):
    x = bank["tiles"]
    n, steps, width = x.shape
    scale = StandardScaler().fit(x[draw].reshape(-1, width))
    x = scale.transform(x.reshape(-1, width)).reshape(n, steps, width)
    shuffled = shuffle_tiles(x, seed + 73 + fold)
    result = {name: fit_predict(lambda: PatchNet(width), (z[draw],), (z[held],), y[draw], seed+fold)
              for name, z in (("ordered", x), ("shuffled", shuffled))}
    result["pooled"] = softmax(logits(x.mean(axis=1), draw, held, y), axis=1)
    return result


def shuffle_partners(people, interface, y, seed):
    rng = np.random.default_rng(seed)
    cells = np.column_stack((people, interface, y))
    _, group = np.unique(cells, return_inverse=True, axis=0)
    shuffled = np.arange(len(y))
    for key in np.unique(group):
        rows = np.flatnonzero(group == key)
        if len(rows) > 1:
            order = rng.permutation(rows)
            shuffled[order] = np.roll(order, 1)
    return shuffled


def bilinear(bank, draw, held, y, people, interface, seed, fold):
    a, b = (bank[key] for key in ("rank", "temporal"))
    views = [StandardScaler().fit(x[draw]).transform(x) for x in (a, b)]
    train = tuple(x[draw] for x in views)
    test = tuple(x[held] for x in views)
    shuffle = shuffle_partners(people[draw], interface[draw], y[draw], seed+fold+408)
    result = {}
    for name, product, data in (("product", True, train), ("additive", False, train),
                                ("shuffled", True, (train[0], train[1][shuffle]))):
        result[name] = fit_predict(lambda: CrossView(a.shape[1], b.shape[1], product),
                                   data, test, y[draw], seed+fold)
    result["temporal"] = softmax(logits(b, draw, held, y), axis=1)
    return result
