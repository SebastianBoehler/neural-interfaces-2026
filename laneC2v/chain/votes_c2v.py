"""Lane C2v: calibrate what existing expert votes (C1, raw-FBC slot) add to the terminal decision. Emission-only post-edit, offline on stored terminals.
usage: votes_c2v.py BASE   (BASE = hardcb | ds_l5)  -> cells_BASE.json {order: {arm: {person: [sum cell BA, cells]}}} + timing. Registered in evidence/wave_20nov/laneC2v_votes.md."""
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

WORK = Path(__file__).resolve().parent
DATA = Path(os.environ.get("C2V_DATA", WORK / "data"))
ORDERS = ("full0", "within0")
GAPS = (0.25, 0.65)
GRID = [(k, c, d) for k in (20, 40, 100) for c in (np.log(1.5), np.log(2.0)) for d in (0.5, 1.0)]
PRIMARY = (40, np.log(2.0), 0.5)


def log_post(T):
    """terminal rows are probabilities or unnormalised log scores (all <= 0); both -> normalised log posterior."""
    isp = T.min(1) >= 0
    assert np.all(np.abs(T[isp].sum(1) - 1) < 1e-6) and np.all(T[~isp] <= 0)  # log rows are unnormalised log scores, renormalised here
    lp = np.where(isp[:, None], np.log(np.clip(T, 1e-12, None)), T)
    return lp - np.logaddexp.reduce(lp, axis=1, keepdims=True)


class Tape:
    """per-window arrays: top-two classes a,b of the terminal, log-odds gap g, gap bin, joint vote tuple v, label y, weight w, log terminal lp."""
    FIELDS = ("a", "b", "g", "bin", "v", "y", "w", "lp")

    def __init__(self, T, V, y, w):
        o = np.argsort(-log_post(T), axis=1)
        self.lp = log_post(T)
        r = np.arange(len(T))
        self.a, self.b = o[:, 0], o[:, 1]
        self.g = self.lp[r, self.a] - self.lp[r, self.b]
        self.bin = (self.a * 3 + self.b) * 3 + np.digitize(self.g, GAPS)
        self.v, self.y, self.w = V, y, w

    def sub(self, ix):
        s = object.__new__(Tape)
        for k in self.FIELDS:
            setattr(s, k, getattr(self, k)[ix])
        return s

    @staticmethod
    def cat(parts):
        s = object.__new__(Tape)
        for k in Tape.FIELDS:
            setattr(s, k, np.concatenate([getattr(q, k) for q in parts]))
        return s


def weights(y, groups):
    key = np.array([f"{s}|{x}|{i}|{c}" for (s, x, i), c in zip(map(tuple, groups), y)])
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]
    return w * len(w) / w.sum()


def fit(tr, K, perm_seed=None):
    """tr: training Tape. Returns (L0[27] blind logit, L1[27,9] vote-conditional shrunk logit, mean gap[27])."""
    inb = (tr.y == tr.a) | (tr.y == tr.b)
    isa, v = tr.y == tr.a, tr.v.copy()
    if perm_seed is not None:  # vote tuples permuted within each pair/gap bin
        rng = np.random.default_rng(perm_seed)
        for b in np.unique(tr.bin):
            ix = np.flatnonzero(tr.bin == b)
            v[ix] = v[rng.permutation(ix)]
    A, B, G, Wt = np.zeros(27), np.zeros(27), np.zeros(27), np.zeros(27)
    Ca, Cb = np.zeros((27, 9)), np.zeros((27, 9))
    ma, mb = inb & isa, inb & ~isa
    np.add.at(G, tr.bin, tr.w * tr.g); np.add.at(Wt, tr.bin, tr.w)
    np.add.at(A, tr.bin[ma], tr.w[ma]); np.add.at(B, tr.bin[mb], tr.w[mb])
    np.add.at(Ca, (tr.bin[ma], v[ma]), tr.w[ma]); np.add.at(Cb, (tr.bin[mb], v[mb]), tr.w[mb])
    pb = (A + .5) / (A + B + 1)
    L1 = np.log((Ca + K * pb[:, None] + 1e-9) / (Cb + K * (1 - pb)[:, None] + 1e-9))
    return np.log((A + .5) / (B + .5)), L1, G / np.maximum(Wt, 1e-9)


def apply(te, tab, kind, clip, dose):
    L0, L1, mg = tab
    if kind == "joint":
        d = L1[te.bin, te.v] - L0[te.bin]
    elif kind == "blind":
        d = L0[te.bin] - mg[te.bin]
    else:
        d = np.zeros(len(te.g))
    d = np.clip(d, -clip, clip) * dose
    lp, r = te.lp.copy(), np.arange(len(te.g))
    lp[r, te.a] += d; lp[r, te.b] -= d
    return lp - np.logaddexp.reduce(lp, axis=1, keepdims=True)  # log posterior


def lopo(tapes, person, src, tgt, kind, K, clip, dose, perm=None):
    """edit order tgt; the table of person p is fitted on source orders `src` with p excluded."""
    out = np.zeros_like(tapes[tgt]["T"])  # log posteriors
    for p in sorted(set(person)):
        te = np.flatnonzero(person == p)
        tr = Tape.cat([tapes[o]["tape"].sub(np.flatnonzero(person != p)) for o in src])
        out[te] = apply(tapes[tgt]["tape"].sub(te), fit(tr, K, perm), kind, clip, dose)
    return out


def main(base):
    from bridge_owner import pin, start_module
    from complete_score import METRIC, METRIC_HASH
    from complete_zero import FEAT, HASHES
    pin(METRIC, METRIC_HASH); pin(FEAT, HASHES[FEAT])
    metric = start_module("final_official_metric", METRIC, METRIC.read_text())
    with np.load(FEAT, allow_pickle=False) as z:
        y, subjects, sessions, interface = (z[k] for k in ("y", "subject", "session", "interface"))
    groups = np.stack((subjects, sessions, interface), axis=1)
    person = np.asarray([p.rsplit("/", 1)[-1] for p in subjects])
    people = sorted(set(person))
    w = weights(y, groups)
    tapes = {}
    for o in ORDERS:
        with np.load(DATA / f"{o}_P.npz", allow_pickle=False) as P, np.load(DATA / f"{o}_w1_0_4.npz", allow_pickle=False) as R:
            assert (R["y"] == y).all() and (R["person"] == person).all() and (R["delivery"] == P["delivery"]).all()
            T = P[f"{base}|terminal"].astype(np.float64)
            V = R["member|c1"].argmax(1) * 3 + R["member|sc_fbraw"].argmax(1)
        tapes[o] = dict(T=T, tape=Tape(T, V, y, w))

    def score(post):
        g = post.argmax(-1)
        s = {p: metric.group_scores(metric.balanced_accuracy, y[person == p], g[person == p], groups[person == p]) for p in people}
        return {p: [float(v.sum()), int(len(v))] for p, v in s.items()}

    def total(c):
        return sum(v[0] for v in c.values()) / sum(v[1] for v in c.values())
    cells, timing = {}, {}
    for o, oth in (("full0", "within0"), ("within0", "full0")):
        arms = {"base": tapes[o]["T"]}
        arms["zero"] = lopo(tapes, person, (oth,), o, "zero", *PRIMARY)
        np.testing.assert_allclose(arms["zero"], log_post(arms["base"]), atol=1e-9)  # zero-change guard (log/renorm round trip only)
        np.testing.assert_array_equal(arms["zero"].argmax(1), arms["base"].argmax(1))
        for sn, src in {"other": (oth,), "same": (o,), "both": ORDERS}.items():
            for kind in ("blind", "joint"):
                t0 = perf_counter()
                arms[f"{kind}|{sn}"] = lopo(tapes, person, src, o, kind, *PRIMARY)
                timing[f"{o}|{kind}|{sn}"] = 1e3 * (perf_counter() - t0) / len(y)
            for s in range(3):
                arms[f"perm{s}|{sn}"] = lopo(tapes, person, src, o, "joint", *PRIMARY, perm=s)
        cb = total(score(tapes[oth]["T"]))  # selection on the OTHER order only (nested LOPO there)
        sel = [total(score(lopo(tapes, person, (oth,), oth, "joint", K, c, d))) - cb for (K, c, d) in GRID]
        best = GRID[int(np.argmax(sel))]
        arms["joint_sel|other"] = lopo(tapes, person, (oth,), o, "joint", *best)
        cells[o] = {k: score(v) for k, v in arms.items()}
        cells[o]["_selected"] = dict(grid=[[float(a), float(b), float(c)] for a, b, c in GRID], delta_other_order=sel, chosen=[float(x) for x in best])
        print(o, base, {k: round(100 * (total(v) - total(cells[o]["base"])), 3) for k, v in cells[o].items() if k != "_selected"}, "chosen", best, flush=True)
    (WORK / f"cells_{base}.json").write_text(json.dumps(dict(base=base, cells=cells, ms_per_window=timing, scope="DEV ten reused held-new people, emission-only post-edit"), indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
