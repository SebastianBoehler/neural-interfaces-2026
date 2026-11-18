"""Identical disjoint visit boundaries and deterministic shuffled delivery."""
from pathlib import Path
import numpy as np


def load(person, smoke):
    path = Path('smoke.npz' if smoke else f'sub-{person:02d}.npz')
    with np.load(path) as z:
        x, names = z['x'], z['names'].tolist()
    assert x.shape == (1 if smoke else 3, 18, 41, 480) and x.dtype == np.float32
    return x, names


def orders(person, visit, smoke=False):
    for seed in ((429,) if smoke else (429, 1429)):
        rows = np.random.default_rng([person, visit, seed]).permutation(18)
        yield seed, rows
