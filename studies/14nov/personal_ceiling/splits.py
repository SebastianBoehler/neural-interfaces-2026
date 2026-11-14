"""PERSONAL CEILING splits and scoring (ledger s246). Order-free and dependency-light (numpy only), shared by the CPU
pipelines (classic.py), the GPU fine-tuning (ftnet.py) and the scorer (score.py).
Within (W): per session, test = one pair of online runs (all six pairs), train = the other four runs (both acquisition
runs + two online runs). Cross (X): P04 / P10, train = all of session 1, test = sessions 2 and 3 on the same pair windows.
Score: balanced accuracy per interface cell of the test windows, mean over the cells present."""
from itertools import combinations
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "runs/personal_ceiling"
PEOPLE = [f"P{i:02d}" for i in range(1, 11)]


def meta():
    with np.load(REPO / "data/proteus_windows.npz") as f:
        m = {k: f[k] for k in ("y", "subject", "session", "interface", "task", "run")}
    m["person"] = np.array([s.split("/")[-1] for s in m["subject"]]); m["sess"] = np.char.add(np.char.add(m["person"], "_"), m["session"])
    return m


def sessions(m):
    return list(dict.fromkeys(m["sess"]))


def pairs(m, s):
    """-> [(name, train rows, test rows)] for session s; rows in recording order."""
    rows = np.flatnonzero(m["sess"] == s); onl = sorted({int(r) for r, t in zip(m["run"][rows], m["task"][rows]) if t.startswith("Online")})
    assert len(onl) == 4, (s, onl); out = []
    for a, b in combinations(onl, 2):
        te = np.isin(m["run"][rows], (a, b)); out.append((f"{onl.index(a) + 1}{onl.index(b) + 1}", rows[~te], rows[te]))
    return out  # names: online-run positions, '34' = last two online runs, '12' = online runs 1+2


def cross(m):
    """-> [(person, train session, test session)] for the cross-session part"""
    return [(p, f"{p}_1", f"{p}_{k}") for p in ("P04", "P10") for k in ("2", "3")]


def ba(y, pred):
    rec = [np.mean(pred[y == c] == c) for c in np.unique(y)]
    return float(np.mean(rec))


def cell_score(m, rows, P):
    """mean over interface cells of balanced accuracy; P (len(rows), 3)"""
    y, it, pred = m["y"][rows], m["interface"][rows], np.asarray(P).argmax(1)
    return 100 * float(np.mean([ba(y[it == c], pred[it == c]) for c in np.unique(it)]))


def fuse(*Ps):
    L = sum(np.log(np.clip(np.asarray(P, np.float64), 1e-6, 1)) for P in Ps); L -= L.max(1, keepdims=True); E = np.exp(L)
    return E / E.sum(1, keepdims=True)
