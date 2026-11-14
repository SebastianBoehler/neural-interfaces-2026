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
RUNS = {u: np.flatnonzero(RUN == u) for u in dict.fromkeys(RUN)}  # true runs: capacities and the urn check
RUN_IDS = os.environ.get("RUN_IDS")  # npz, key <scheme><seed>: inferred run label per replay row (the filter then uses those groups)
# by run type, from the public runs' counts: online runs are exactly n/3 per class (30 -> 10, two BrainHero 36-window runs -> 12);
# acquisition runs are a 12/12/12 design (Graz exact; BrainHero 9-12 per class after missing trials)
CAP = {u: (len(r) // 3 if "Online" in u.split("|")[1] else 12) for u, r in RUNS.items()}
for u, r in RUNS.items():  # the urn assumption, checked on the labels once (the filter never sees them)
    assert np.bincount(y[r], minlength=3).max() <= CAP[u], f"run {u} holds more than {CAP[u]} windows of one class"


def rank(order):
    sc, sd = order.rstrip("0123456789"), int(order[len(order.rstrip("0123456789")):]); g = PERMS[f"{sc}|{sd}|{p0}"]
    rk = np.empty(len(y), int); rk[g] = np.arange(len(g)); return rk


def filt(P, rk, groups=None):
    out = P.copy()
    for u, r in (groups or RUNS).items():
        r = r[np.argsort(rk[r])]; cap = CAP.get(u, 12); alpha = {(0, 0, 0): 1.0}  # inferred groups: the 12/12/12 bound
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


print(f"RUNCOUNT (s215): TAU {TAU}, FLOOR {FLOOR}, true run ids; change against offsets off / filter off, new people"); SAVE = {}
for order in sys.argv[1:]:
    rk = rank(order); res = {}
    G = None
    if RUN_IDS:  # inferred runs (per session): groups of replay rows sharing a session and an inferred label
        lab = np.char.add(ses.astype(str), np.char.add("|", np.load(RUN_IDS)[order].astype(str))); G = {u: np.flatnonzero(lab == u) for u in dict.fromkeys(lab)}
    for v in ("off", "on"):
        z = np.load(f"{REPO}/runs/chainout/s215_{order}_{v}.npz"); bases = sorted({k.split("|")[0] for k in z.files})
        for arm in ("P0r", "V6Fr"):
            for fl in (0, 1):
                outs = [z[f"{bn}|{arm}"].astype(np.float64) for bn in bases]; outs = [filt(o, rk, G) if fl else o for o in outs]
                res[v, arm, fl] = (np.mean([SC.person(o, p0) for o in outs], 0), np.mean([clustering(o) for o in outs]))
    SAVE[order] = {f"{v}|{arm}|{fl}": res[v, arm, fl][0] for (v, arm, fl) in res}
    for v in ("off", "on"):  # sharpness of the unfiltered output the filter's temperature acts on (s215 detail)
        z = np.load(f"{REPO}/runs/chainout/s215_{order}_{v}.npz"); bases = sorted({k.split("|")[0] for k in z.files})
        SAVE[order][f"sharp|{v}"] = np.array([np.mean([z[f"{bn}|{arm}"].max(1).mean() for bn in bases]) for arm in ("P0r", "V6Fr")])
    for arm in ("P0r", "V6Fr"):
        b = res["off", arm, 0][0]
        cells = " | ".join(f"offsets {v} filter {fl}: {(res[v, arm, fl][0] - b).mean():+.2f} (clust {res[v, arm, fl][1]:.2f})" for v in ("off", "on") for fl in (0, 1))
        print(f"  {order:8s} {arm:4s} {cells} | filter alone without P10 {np.delete(res['off', arm, 1][0] - b, 9).mean():+.2f}", flush=True)

if os.environ.get("DETAIL"):  # per-person cells, 80 % person intervals, sharpness; saved for later reading
    rng = np.random.default_rng(0); tag = os.path.basename(RUN_IDS)[:-4] if RUN_IDS else "true"
    np.savez(f"{REPO}/runs/s215_detail_{tag}_tau{TAU:g}.npz", **{f"{o}|{k}": v for o, d in SAVE.items() for k, v in d.items()})
    print("\nPer order, v6f / v5.1b: cell - (offsets off, filter off), mean [80 % person interval]; sharpness = mean top posterior off -> on")
    for o, d in SAVE.items():
        for j, arm in enumerate(("P0r", "V6Fr")):
            b = d[f"off|{arm}|0"]; cells = []
            for v, fl in (("off", 1), ("on", 0), ("on", 1)):
                x = d[f"{v}|{arm}|{fl}"] - b; bs = [x[rng.integers(0, 10, 10)].mean() for _ in range(2000)]
                cells.append(f"{'filter' if (v, fl) == ('off', 1) else 'offsets' if fl == 0 else 'both'} {x.mean():+.2f} [{np.percentile(bs, 10):+.2f}, {np.percentile(bs, 90):+.2f}]")
            print(f"  {o:8s} {arm:4s} " + " | ".join(cells) + f" | sharpness {d['sharp|off'][j]:.3f} -> {d['sharp|on'][j]:.3f}")
