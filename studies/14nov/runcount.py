"""RUNCOUNT (ledger s215, panel round 2 / statistician): a run-level count filter on the chain's final posteriors, with
TRUE run membership. Every public run is class-balanced (online runs 10/10/10; acquisition runs a 12/12/12 design, some
with missing trials, so at most 12 per class). Each run is an urn without replacement: an exact forward filter over
the class counts seen so far, in delivery order, with the chain's posterior^TAU as the likelihood and a soft floor (a class
keeps FLOOR pseudo-items, so one wrong boundary cannot zero it). The output for a window uses only windows delivered up
to it (causal). Gates and fine-tuning labels are untouched: this acts on the emitted output only.
Input: shufcombo SAVE_OUT files (runs/chainout/s215_<order>_<off|on>.npz: <base>|<arm> -> (N, 3), recording layout).
Reports per order the 2 x 2 (run offsets off / on x filter off / on) for v5.1b (P0r) and v6f (V6Fr), new people, and the
run-clustering ratio of the errors (variance of the per-run error rate over its binomial expectation) before and after.
Usage: [TAU=2] [FLOOR=0.5] python studies/14nov/runcount.py order [order ...]"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../.."); sys.path.insert(0, REPO + "/replay")
import env  # noqa: E402
A = env.A; SC = A.SC; p0 = "all_loso"; Z = A.L.Z; IDX = A.pl.B[f"idx|{p0}"].astype(int)
y, ses = Z[f"y|{p0}"].astype(int), Z[f"sess|{p0}"]; PERMS = np.load(REPO + "/runs/shuffle_perms.npz")
TAU, FLOOR = float(os.environ.get("TAU", 2)), float(os.environ.get("FLOOR", 0.5))
with np.load(REPO + "/data/proteus_windows.npz") as w:
    RUN = np.char.add(ses.astype(str), np.char.add("|", np.char.add(w["task"].astype(str), np.char.add("|", w["run"].astype(str)))[IDX]))
RUNS = {u: np.flatnonzero(RUN == u) for u in dict.fromkeys(RUN)}
# by run type, from the public runs' counts: online runs are exactly n/3 per class (30 -> 10, two BrainHero 36-window runs -> 12);
# acquisition runs are a 12/12/12 design (Graz exact; BrainHero 9-12 per class after missing trials)
CAP = {u: (len(r) // 3 if "Online" in u.split("|")[1] else 12) for u, r in RUNS.items()}
for u, r in RUNS.items():  # the urn assumption, checked on the labels once (the filter never sees them)
    assert np.bincount(y[r], minlength=3).max() <= CAP[u], f"run {u} holds more than {CAP[u]} windows of one class"


def rank(order):
    sc, sd = order.rstrip("0123456789"), int(order[len(order.rstrip("0123456789")):]); g = PERMS[f"{sc}|{sd}|{p0}"]
    rk = np.empty(len(y), int); rk[g] = np.arange(len(g)); return rk


def filt(P, rk):
    out = P.copy()
    for u, r in RUNS.items():
        r = r[np.argsort(rk[r])]; cap = CAP[u]; alpha = {(0, 0, 0): 1.0}
        for t, i in enumerate(r):
            L = np.clip(P[i], 1e-9, 1) ** TAU; L /= L.sum(); new, pred = {}, np.zeros(3)
            for k, a in alpha.items():
                tr = np.array([max(cap - k[c], 0) + FLOOR for c in range(3)]); tr /= tr.sum()
                for c in range(3):
                    v = a * tr[c] * L[c]; pred[c] += v; kk = k[:c] + (k[c] + 1,) + k[c + 1:]; new[kk] = new.get(kk, 0.0) + v
            s = sum(new.values()); alpha = {k: v / s for k, v in new.items()}; out[i] = pred / pred.sum()
    return out


def clustering(P):
    e = (P.argmax(1) != y).astype(float); p = e.mean(); rates = [e[r].mean() for r in RUNS.values()]; n = np.mean([len(r) for r in RUNS.values()])
    return np.var(rates) / (p * (1 - p) / n)


print(f"RUNCOUNT (s215): TAU {TAU}, FLOOR {FLOOR}, true run ids; change against offsets off / filter off, new people")
for order in sys.argv[1:]:
    rk = rank(order); res = {}
    for v in ("off", "on"):
        z = np.load(f"{REPO}/runs/chainout/s215_{order}_{v}.npz"); bases = sorted({k.split("|")[0] for k in z.files})
        for arm in ("P0r", "V6Fr"):
            for fl in (0, 1):
                outs = [z[f"{bn}|{arm}"].astype(np.float64) for bn in bases]; outs = [filt(o, rk) if fl else o for o in outs]
                res[v, arm, fl] = (np.mean([SC.person(o, p0) for o in outs], 0), np.mean([clustering(o) for o in outs]))
    for arm in ("P0r", "V6Fr"):
        b = res["off", arm, 0][0]
        cells = " | ".join(f"offsets {v} filter {fl}: {(res[v, arm, fl][0] - b).mean():+.2f} (clust {res[v, arm, fl][1]:.2f})" for v in ("off", "on") for fl in (0, 1))
        print(f"  {order:8s} {arm:4s} {cells} | filter alone without P10 {np.delete(res['off', arm, 1][0] - b, 9).mean():+.2f}", flush=True)
