"""Cell/class-balanced diagnostic breakdown, checked against official accuracy."""
import numpy as np


def probability(p):
    p = np.asarray(p)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError('invalid probability shape or finite values')
    if np.any(p < -1e-7) or np.any(p > 1 + 1e-7) or not np.allclose(p.sum(1), 1, atol=1e-6, rtol=0):
        raise ValueError('stage output is not a probability distribution')
    return p


def values_by_person(values, y, cell, person, people):
    values, y, cell, person = map(np.asarray, (values, y, cell, person))
    if values.shape != y.shape or cell.shape != y.shape or person.shape != y.shape:
        raise ValueError('cell layout mismatch')
    out = {p: [] for p in people}
    for c in np.unique(cell):
        rows = cell == c; owners = np.unique(person[rows])
        if len(owners) != 1 or owners[0] not in out:
            raise ValueError('cell has invalid owner')
        classes = np.unique(y[rows])
        out[owners[0]].append(np.mean([values[rows & (y == k)].mean() for k in classes]))
    if any(not v for v in out.values()): raise ValueError('missing person')
    return np.array([np.mean(out[p]) for p in people])


def transitions(before, after, y, cell, person, people):
    before, after, y, cell, person = map(np.asarray, (before, after, y, cell, person))
    up = np.zeros((len(people), 3)); down = np.zeros_like(up); confusion = np.zeros((len(people), 3, 3))
    lookup = {p: i for i, p in enumerate(people)}; counts = np.zeros(len(people))
    for c in np.unique(cell):
        rows = cell == c; owners = np.unique(person[rows])
        if len(owners) != 1 or owners[0] not in lookup: raise ValueError('invalid cell owner')
        i = lookup[owners[0]]; counts[i] += 1; classes = np.unique(y[rows])
        for k in classes:
            q = rows & (y == k); weight = 100. / len(classes)
            up[i, k] += weight * np.mean((before[q] != k) & (after[q] == k))
            down[i, k] += weight * np.mean((before[q] == k) & (after[q] != k))
            for j in range(3): confusion[i, k, j] += weight * np.mean(after[q] == j)
    if np.any(counts == 0): raise ValueError('missing person')
    return up / counts[:, None], down / counts[:, None], confusion / counts[:, None, None]


def calibration(p, y, cell, person, people):
    p = probability(p); y = np.asarray(y, int)
    if np.any((y < 0) | (y > 2)): raise ValueError('invalid class')
    true = p[np.arange(len(y)), y]
    others = p.copy(); others[np.arange(len(y)), y] = -np.inf
    brier = np.sum((p - np.eye(3)[y]) ** 2, axis=1)
    margin = true - others.max(axis=1)
    return (values_by_person(brier, y, cell, person, people),
            values_by_person(margin, y, cell, person, people))
