"""NI-V delivery schedules for Scherer 2015 (9 people x 2 visits): full-session and within-run shuffle.

full   : every window of a visit is shuffled (sealed-like "full-session shuffle").
within : windows are shuffled inside each acquisition run, runs keep their chronological order.
The permutation function is the one of studies/17nov/external_pair352/data.py (same RNG stream per scheme id).
"""
import gzip
import csv
import hashlib

import numpy as np

SCHEME_ID = {"full": 0, "runs": 1, "within": 2}
SEEDS = tuple(71000 + k for k in range(8))          # eight pinned orders per scheme
SCHEMES = ("full", "within")


def stamp(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def permutation(stream, group, scheme, seed):
    rng = np.random.default_rng([seed, 0, SCHEME_ID[scheme]])
    p = np.arange(len(stream))
    for session in dict.fromkeys(stream):
        rows = np.flatnonzero(stream == session)
        if scheme == "full":
            p[rows] = rows[rng.permutation(len(rows))]
        else:
            blocks = [rows[group[rows] == g] for g in dict.fromkeys(group[rows])]
            if scheme == "runs":
                blocks = [blocks[i] for i in rng.permutation(len(blocks))]
            else:
                blocks = [b[rng.permutation(len(b))] for b in blocks]
            p[rows] = np.concatenate(blocks)
    assert np.array_equal(np.sort(p), np.arange(len(stream)))
    return p


def runs_from_manifest(path, subject, session):
    """Acquisition run of every Scherer row (the npz run column is all zero; the row manifest has the real runs)."""
    with gzip.open(path, "rt") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert len(rows) == len(subject) and [int(r["row"]) for r in rows] == list(range(len(rows)))
    for i, r in enumerate(rows):
        assert subject[i] == f"Scherer2015/{r['person']}" and session[i] == r["session"]
    return np.array([int(r["run"]) for r in rows])


def plans(subject, session, run, scheme, seed):
    """{visit key 'Scherer2015/G|1': delivered row indices} for one (scheme, order seed)."""
    stream = np.char.add(np.char.add(subject, "|"), session)
    p = permutation(stream, run, scheme, seed)
    return {s: p[stream[p] == s] for s in sorted(set(stream))}
