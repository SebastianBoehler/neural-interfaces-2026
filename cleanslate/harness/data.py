"""Shared data access, delivery orders and official scoring for the clean-slate harness.

Rows are those of data/proteus_windows.npz (2,655 windows, 10 people, 14 sessions, float32 (41,480) at 120 Hz).
Streams = participant/session, cells = stream x interface (official `balanced_accuracy_cell_mean`).
Delivery = full-session shuffle: every window of a session is permuted (rng [seed,0,0], sessions in data order,
same recipe as niv/orders.py scheme "full"), so neither recording order nor interface blocks survive.
Order seeds 72000-72999 select "within" (runs in acquisition order, windows shuffled inside each run), 73000-73999
"runs" (runs in random order, recording order inside) and 74000-74999 "cells" (one block per interface in order of
first acquisition, shuffled inside), with niv/orders.py's RNG stream per scheme. The sealed shuffle scope is not
stated by the organisers, so a stack chosen on full-session orders needs a within-run check too.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

DATA = os.environ.get("CS_DATA", "/data/proteus_windows.npz")
OFFICIAL = os.environ.get("CS_OFFICIAL", "/official/tracks/bci_decoding")
PEOPLE = [f"P{k:02d}" for k in range(1, 11)]
RETURNING = ("P04", "P10")           # the only public people with more than one session
DEFAULT_ORDERS = (71000, 71001)


def official():
    """Official metric code (read-only submodule)."""
    if OFFICIAL not in sys.path:
        sys.path.insert(0, OFFICIAL)
    from benchmark_utils import metrics, streaming
    return metrics, streaming


def load(path=None, mmap=False):
    with np.load(path or DATA, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["person"] = np.array([s.split("/")[-1] for s in d["subject"]])
    d["stream"] = np.char.add(np.char.add(d["person"], "|"), d["session"])
    return d


def train_rows(d, people, first_session_only=()):
    """Row indices a team may train on: all windows of `people`, only session 1 for people in first_session_only."""
    people, first = set(people), set(first_session_only)
    keep = np.isin(d["person"], list(people))
    keep &= ~(np.isin(d["person"], list(first)) & (d["session"] != "1"))
    return np.flatnonzero(keep)


def load_train(people, first_session_only=(), path=None):
    """(X float32 (n,41,480), y, info dict) of the permitted training windows; the helper every train.py should use."""
    d = load(path)
    r = train_rows(d, people, first_session_only)
    return d["X"][r], d["y"][r], {k: d[k][r] for k in ("person", "session", "interface", "task", "run", "onset")}


def train_parser(desc=""):
    """Common train.py command line: --people --first-session-only --out --data."""
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--people", required=True, help="comma list; only these people's windows may be used")
    p.add_argument("--first-session-only", default="", help="comma list of people limited to their first session")
    p.add_argument("--out", required=True, help="package directory to create (submission.py + weights)")
    p.add_argument("--data", default=None)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    a.people = [s for s in a.people.split(",") if s]
    a.first_session_only = [s for s in a.first_session_only.split(",") if s]
    assert set(a.first_session_only) <= set(a.people) or not a.first_session_only
    return a


SCHEME_ID = {"full": 0, "runs": 1, "within": 2, "cells": 3}     # first three as in niv/orders.py


def scheme_of(seed):
    """Delivery scheme of an order seed: 72xxx within-run shuffle, 73xxx run-order shuffle, 74xxx per-interface
    shuffle, anything else full-session."""
    return {72: "within", 73: "runs", 74: "cells"}.get(int(seed) // 1000, "full")


def delivery(stream, seed, run=None, onset=None, interface=None):
    """Permutation p of all rows: p[stream==s] lists the delivery order of session s under this seed."""
    scheme = scheme_of(seed)
    if scheme != "full" and (run is None or onset is None or (scheme == "cells" and interface is None)):
        raise ValueError(f"order {seed} is a {scheme!r} order: delivery() needs run, onset and interface (use plans())")
    rng = np.random.default_rng([seed, 0, SCHEME_ID[scheme]])
    p = np.arange(len(stream))
    for s in dict.fromkeys(stream):
        rows = np.flatnonzero(stream == s)
        if scheme == "full":
            p[rows] = rows[rng.permutation(len(rows))]
            continue
        acq = rows[np.lexsort((onset[rows], run[rows]))]        # acquisition order: BIDS run number, then onset
        group = interface if scheme == "cells" else run
        blocks = [acq[group[acq] == g] for g in dict.fromkeys(group[acq])]
        if scheme == "runs":
            blocks = [blocks[i] for i in rng.permutation(len(blocks))]
        else:
            blocks = [b[rng.permutation(len(b))] for b in blocks]
        p[rows] = np.concatenate(blocks)
    assert np.array_equal(np.sort(p), np.arange(len(stream)))
    return p


def plans(d, seed, streams):
    p = delivery(d["stream"], seed, d.get("run"), d.get("onset"), d.get("interface"))
    return {s: p[d["stream"][p] == s] for s in streams}


def cell_table(d, pred, rows_mask, order, mode):
    """Official cell metric on the rows in rows_mask (pred: label by row, -1 = missing). Returns list of cell dicts."""
    metrics, _ = official()
    out = []
    key = np.char.add(np.char.add(d["stream"], "|"), d["interface"])
    for k in sorted(set(key[rows_mask])):
        r = np.flatnonzero(rows_mask & (key == k))
        if (pred[r] < 0).any():
            raise RuntimeError(f"missing predictions in cell {k}")
        person, session, iface = k.split("|")
        out.append(dict(mode=mode, order=order, person=person, session=session, interface=iface, n=len(r),
                        ba=100 * metrics.balanced_accuracy(d["y"][r], pred[r])))
    return out
