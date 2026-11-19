"""Build held-source same-task pairs and an exact-marginal derangement."""
import hashlib

import numpy as np
import torch

from anchors import row_sha
from relational import groups, unit


def plan(held, source, people, y, batches, embedding):
    membership = {p: g for g, member in enumerate(groups(held)) for p in member}
    digest = hashlib.sha256()
    schedule, selected = [], []
    moved, count = [], 0
    for step, rows in enumerate(r for epoch in batches for r in epoch):
        entries, literal, sham = [], [], []
        if len(rows) not in (32, 64):
            raise ValueError("original batch does not use 32-window chunks")
        left, right = rows[:32], rows[32:]
        if len(right):
            a, b = people[left[0]], people[right[0]]
            if set(people[left]) != {a} or set(people[right]) != {b}:
                raise ValueError("native chunk crossed enrollment")
            if a != b and membership[a] == membership[b]:
                for label in (0, 1, 2):
                    x, z = (np.asarray(list(dict.fromkeys(chunk[y[chunk] == label])), dtype=np.int64)
                            for chunk in (left, right))
                    n = min(len(x), len(z))
                    if n < 2:
                        continue
                    x, z = x[:n], z[:n]
                    value = (embedding[x] * embedding[z]).sum(1, dtype=np.float32)
                    shuffled = np.roll(value, 1)
                    change = float(np.abs(value-shuffled).min())
                    if change <= 1e-6:
                        continue
                    idx = np.stack([np.argmax(x[:, None] == left[None, :], axis=1),
                                    32 + np.argmax(z[:, None] == right[None, :], axis=1)], axis=1)
                    if not np.array_equal(rows[idx], np.stack((x,z),1)) or not np.array_equal(
                            np.sort(value), np.sort(shuffled)):
                        raise ValueError("source relations changed pair identity or value marginal")
                    digest.update(np.asarray((step,label,n),dtype="<i8").tobytes())
                    digest.update(np.asarray(source[rows[idx]],dtype="<i8").tobytes())
                    digest.update(np.asarray(value,dtype="<f4").tobytes())
                    entries.append(idx)
                    literal.append(value)
                    sham.append(shuffled)
                    count += n
                    moved.append(change)
        if entries:
            selected.append(step)
            schedule.append(tuple(np.concatenate(t) for t in (entries,literal,sham)))
        else:
            schedule.append(None)
    if len(selected) < 6 or len(selected)/len(schedule) < .05 or count < 100:
        raise ValueError("insufficient active genuinely excluded source relations")
    proof = dict(pair_tape=digest.hexdigest(), pair_steps=len(selected), pair_count=count,
                 diagnostic_steps=row_sha(selected[:6]), minimum_target_move=min(moved), marginal_exact=True)
    return schedule, selected[:6], proof


def relation(z, indices):
    normalized = unit(z)
    return (normalized[indices[:, 0]] * normalized[indices[:, 1]]).sum(1)


def to_device(doses, device):
    return [None if v is None else tuple(torch.as_tensor(t, device=device) for t in v) for v in doses]
