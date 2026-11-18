"""Make two different partner maps on the exact same source row occurrences."""
import hashlib
from collections import Counter

import numpy as np


WITHIN = (1, 0, 3, 2)
CROSS = (2, 3, 0, 1)


def make_tape(batches, subject, stream, labels, held, seed):
    rng = np.random.default_rng(20020 + 100 * int(held[1:]) + seed)
    owners, positions = [], []
    step = 0
    for epoch in batches:
        can = []
        for i, rows in enumerate(epoch):
            if len(rows) == 64:
                first, second = rows[:32], rows[32:]
                if len(set(stream[first])) != 1 or len(set(stream[second])) != 1:
                    raise ValueError("the original S1C chunks are not session-pure")
                if subject[first[0]] != subject[second[0]]:
                    for label in (0, 1, 2):
                        groups = []
                        for start in (0, 32):
                            q = rng.permutation(np.flatnonzero(labels[rows[start:start + 32]] == label) + start)
                            _, unique = np.unique(rows[q], return_index=True)
                            groups.append(q[np.sort(unique)])
                        n = min(map(len, groups)) // 2
                        if n:
                            pair = np.column_stack([g[:2 * n].reshape(-1, 2) for g in groups])
                            can.extend((step + i, q) for q in pair)
        quota = sum(map(len, epoch)) // 16
        if quota * 16 != sum(map(len, epoch)) or len(can) < quota:
            raise ValueError(f"exact matched quartets cannot cover one quarter, {len(can)} available/{quota} required")
        selected = sorted(rng.choice(len(can), size=quota, replace=False).tolist())
        for j in selected:
            i, q = can[j]
            owners.append(i)
            positions.append(q)
        step += len(epoch)
    return np.asarray(owners, dtype=np.int32), np.asarray(positions, dtype=np.int16)


def verify_tape(batches, subject, stream, step, slots, labels=None):
    flat = [r for epoch in batches for r in epoch]
    if (step.ndim != 1 or slots.shape != (len(step), 4) or not len(step) or
            step.min() < 0 or step.max() >= len(flat) or (np.diff(step) < 0).any()):
        raise ValueError("missing or invalid source replacement layout")
    bounds = np.searchsorted(step, np.arange(len(flat) + 1))
    pairs = [Counter(), Counter()]
    used = np.zeros(len(subject), dtype=np.int64)
    affiliation = []
    for i, rows in enumerate(flat):
        q = slots[bounds[i]:bounds[i + 1]]
        if len(q) == 0:
            continue
        if (len(rows) != 64 or (q[:, :2] < 0).any() or (q[:, :2] >= 32).any() or
                (q[:, 2:] < 32).any() or (q[:, 2:] >= 64).any() or len(np.unique(q)) != q.size):
            raise ValueError("a source occurrence appears twice or is outside its original two chunks")
        actual = rows[q]
        a, b, c, d = actual.T
        if ((a == b).any() or (c == d).any() or (subject[a] == subject[c]).any() or
                (stream[a] != stream[b]).any() or (stream[c] != stream[d]).any() or
                labels is not None and not (labels[actual] == labels[a, None]).all()):
            raise ValueError("source pairing crossed the registered class/session/person rules")
        np.add.at(used, actual.reshape(-1), 1)
        for kind, order in enumerate((WITHIN, CROSS)):
            donors = actual[:, order]
            if not np.array_equal(np.sort(actual.reshape(-1)), np.sort(donors.reshape(-1))):
                raise ValueError("source occurrence donor exposure differs between treatments")
            p = np.sort(np.stack((a, donors[:, 0], d, donors[:, 3]), axis=1).reshape(-1, 2), axis=1)
            pairs[kind].update(map(tuple, p))
        affiliation.append(actual)
    cursor = 0
    for epoch in batches:
        first, last = bounds[cursor], bounds[cursor + len(epoch)]
        if (last - first) * 16 != sum(map(len, epoch)):
            raise ValueError("source epoch mix dose changed")
        cursor += len(epoch)
    if set(pairs[0]) & set(pairs[1]) or not affiliation or len(affiliation[0]) == 0:
        raise ValueError("source partner identities do not distinguish the comparisons")
    total = sum(map(len, flat))
    proof = dict(train_rows_seen=total, matched_rows=len(step) * 4, distinct_within_pairs=len(pairs[0]),
                 distinct_cross_pairs=len(pairs[1]), max_within_pair_repeats=max(pairs[0].values()),
                 max_cross_pair_repeats=max(pairs[1].values()), true_donor_rows=int(np.count_nonzero(used)),
                 max_true_donor_reuse=int(used.max()), donor_reuse_hash=hashlib.sha256(used.astype("<i8").tobytes()).hexdigest(),
                 tape=hashlib.sha256(step.astype("<i4").tobytes() + slots.astype("<i2").tobytes()).hexdigest())
    return bounds, np.concatenate(affiliation, axis=0), proof
