"""Evaluation protocols on the exported PROTEUS windows.

Scores use the official metric helpers. A cell is participant x session x
interface; the ranking metric is the equal-weight mean of cell balanced
accuracies. Models see test windows one session at a time, in acquisition
order, and must be causal within the session (see ``models.py``).

Protocols
---------
official : official train split -> official test split (P04, P09).
official_tv : official train + validation split -> official test split.
dev_loso : leave one participant out over the 8 non-test participants.
           This is the model-selection protocol; it never sees P04/P09.
dev_session : train on all dev windows except P10 sessions 2-3, test on
              them (a small probe of later-session transfer).
all_loso : leave one participant out over all 10 participants. Includes the
           official test participants, so it only *evaluates* candidates
           already chosen on the dev protocols; never select on it.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "2026-competition/tracks/bci_decoding"))

from benchmark_utils.metrics import balanced_accuracy, group_scores  # noqa: E402

DATA = ROOT / "data/proteus_windows.npz"
PROTOCOLS = ("official", "official_tv", "dev_loso", "dev_session", "all_loso")


def load():
    with np.load(DATA) as f:
        d = {k: f[k] for k in f.files}
    d["stream"] = np.char.add(np.char.add(d["subject"], "|"), d["session"])
    return d


def folds(d, protocol):
    """Yield ``(name, train_mask, test_mask)`` for a protocol."""
    split, subj = d["split"], d["subject"]
    if protocol == "official":
        yield "official", split == "train", split == "test"
    elif protocol == "official_tv":
        yield "official_tv", split != "test", split == "test"
    elif protocol == "dev_loso":
        dev = split != "test"
        for s in sorted(set(subj[dev])):
            yield s.split("/")[-1], dev & (subj != s), dev & (subj == s)
    elif protocol == "all_loso":
        for s in sorted(set(subj)):
            yield s.split("/")[-1], subj != s, subj == s
    elif protocol == "dev_session":
        dev = split != "test"
        held = (subj == "Dreyer2026Proteus/P10") & (d["session"] != "1")
        yield "P10_s23", dev & ~held, held
    else:
        raise ValueError(protocol)


def score(y, pred, d, mask):
    cells = np.stack([d["stream"][mask], d["interface"][mask]], axis=1)
    cell = group_scores(balanced_accuracy, y, pred, cells)
    sess = group_scores(balanced_accuracy, y, pred, d["stream"][mask])
    return {
        "balanced_accuracy_cell_mean": float(cell.mean()),
        "balanced_accuracy_session_mean": float(sess.mean()),
        "balanced_accuracy": balanced_accuracy(y, pred),
        "accuracy": float(np.mean(y == pred)),
        "n_cells": int(len(cell)),
        "cell_scores": [float(c) for c in cell],
    }


def predict_streams(model, d, mask):
    """Predict test windows session by session, in stored acquisition order."""
    idx = np.flatnonzero(mask)
    pred = np.empty(len(idx), dtype=np.int64)
    streams = d["stream"][idx]
    start = 0
    while start < len(idx):
        stop = start
        while stop < len(idx) and streams[stop] == streams[start]:
            stop += 1
        assert streams[start] not in streams[:start], "stream not contiguous"
        pred[start:stop] = model.predict_stream(idx[start:stop])
        start = stop
    return pred


def evaluate(make_model, d, protocol):
    """Run every fold; returns pooled scores plus per-fold scores."""
    ys, preds, masks, per_fold = [], [], [], {}
    for name, tr, te in folds(d, protocol):
        model = make_model()
        model.fit(np.flatnonzero(tr))
        pred = predict_streams(model, d, te)
        y = d["y"][te]
        per_fold[name] = score(y, pred, d, te)["balanced_accuracy_cell_mean"]
        ys.append(y), preds.append(pred), masks.append(te)
    mask = np.any(masks, axis=0)
    order = np.argsort(np.concatenate([np.flatnonzero(m) for m in masks]))
    y = np.concatenate(ys)[order]
    pred = np.concatenate(preds)[order]
    out = score(y, pred, d, mask)
    out["per_fold"] = per_fold
    return out
