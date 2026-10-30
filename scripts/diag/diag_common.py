"""Shared loaders for the R15 probability diagnostics (saved probabilities only)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "scripts")]
import numpy as np
import protocol
from mix_probs import member, parse
from ensemble import rebalance
from sklearn.metrics import balanced_accuracy_score as bacc

DIRS = ("probs_r15ss,probs_r14,probs_shin,probs_r11,probs_r10,probs_r9,probs,probs_cloud,probs_r5,probs_local,"
        "probs_r7,probs_tf,probs_cloud_r8").split(",")
# R15 six-seed composition (TF members 0-4), as in scratchpad/seeds6_compare.py
R15 = ["v2", "fbcnet_ea_tf:0-4*0.5", "fbcnet_raw_tf:0-4*0.5", "shallow_ea:0-5", "shallow_raw:0-5",
       "fbcnet_raw:0-5", "fbcnet_ea_rot05:0-5", "sincshallow_ea_ptm7r_rot05:0-5", "fblight_ea_rot05:0-5"]
# seen_session has no rot05 / ptm7r files: R10-like proxy with the non-rotation EA members
PROXY = ["v2", "fbcnet_ea_tf:0-4*0.5", "fbcnet_raw_tf:0-4*0.5", "shallow_ea:0-5", "shallow_raw:0-5",
         "fbcnet_raw:0-5", "fbcnet_ea:0-5", "sincshallow_ea_ptm7:0-5", "fblight_ea:0-5"]
FAM = {"v2": "v2", "fbcnet_ea_tf": "fbcnet_tf", "fbcnet_raw_tf": "fbcnet_tf", "shallow_ea": "shallow",
       "shallow_raw": "shallow", "fbcnet_raw": "fbcnet", "fbcnet_ea_rot05": "fbcnet", "fbcnet_ea": "fbcnet",
       "sincshallow_ea_ptm7r_rot05": "sinc", "sincshallow_ea_ptm7": "sinc",
       "fblight_ea_rot05": "fblight", "fblight_ea": "fblight"}
INP = {n: ("v2" if n == "v2" else "raw" if "raw" in n else "ea") for n in FAM}
d = protocol.load()
_cache = {}


def load(proto, specs=None):
    """-> names, list of P (N,3), weights, idx; rows sorted by idx (= acquisition order)."""
    specs = specs or (PROXY if proto == "seen_session" else R15)
    key = (proto, tuple(specs))
    if key in _cache:
        return _cache[key]
    src = "all_loso" if proto == "dev_loso_all" else proto
    names, Ps, ws, seeds, idx = [], [], [], {}, None
    for n, s, w in map(parse, specs):
        P, ix, sd = member(n, s, src, DIRS)
        o = np.argsort(ix)
        P, ix = P[o], ix[o]
        assert idx is None or (ix == idx).all()
        idx = ix
        names.append(n); Ps.append(P); ws.append(w); seeds[n] = sd
    if proto == "dev_loso_all":
        keep = ~np.isin(d["subject"][idx], [f"Dreyer2026Proteus/{p}" for p in ("P04", "P09")])
        Ps = [P[keep] for P in Ps]; idx = idx[keep]
    _cache[key] = (names, Ps, np.array(ws), idx, seeds)
    return _cache[key]


def mixp(Ps, ws):
    ws = np.asarray(ws, float)
    return sum(w / ws.sum() * P for w, P in zip(ws, Ps))


def rebal(P, idx, alpha=1.0, k=5.0, W=30):
    st = d["stream"][idx]
    out = np.empty_like(P)
    for s in dict.fromkeys(st):
        r = st == s
        out[r] = rebalance(P[r], alpha, k, W) if alpha else P[r]
    return out


def cells(idx, pred):
    """{(stream, interface): balanced accuracy %}."""
    st, it, y = d["stream"][idx], d["interface"][idx], d["y"][idx]
    out = {}
    for s in dict.fromkeys(st):
        for i in ("Graz", "BrainHero"):
            m = (st == s) & (it == i)
            if m.any():
                out[(s, i)] = 100 * bacc(y[m], pred[m])
    return out


def cm(idx, pred):
    return float(np.mean(list(cells(idx, pred).values())))


def short(s):
    return s.split("/")[-1]


def position(idx):
    """Position of each row within its stream and within its run."""
    st, run = d["stream"][idx], d["run"][idx]
    pos, rpos, rk = np.zeros(len(idx), int), np.zeros(len(idx), int), np.zeros(len(idx), int)
    for s in dict.fromkeys(st):
        r = np.flatnonzero(st == s)
        pos[r] = np.arange(len(r))
        for j, ru in enumerate(dict.fromkeys(run[r])):
            q = r[run[r] == ru]
            rpos[q] = np.arange(len(q)); rk[q] = j
    return pos, rpos, rk
