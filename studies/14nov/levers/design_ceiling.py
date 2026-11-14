"""s248 M1: ceiling of perfect design knowledge on the chain's final posteriors (new people, all_loso rows).
Arms, change against the raw output (argmax), per order, mean over the six bases:
  b  causal true-run count filter (runcount.filt, TAU 2, FLOOR 0.5, caps 10 online / 12 acquisition): the s215 reference
  e  the same causal filter with each run's TRUE class counts as capacities (design knowledge beyond boundaries)
  c  offline exact-count assignment per run (Hungarian on log posteriors, true counts): non-causal upper bound
  d  offline exact counts per cell (session x interface), f  offline exact counts per session
Usage: python studies/14nov/levers/design_ceiling.py within0 within1 full0 full1 runs0 runs1"""
import os, sys
import numpy as np
from scipy.optimize import linear_sum_assignment
REPO = os.path.abspath(__file__ + "/../../../.."); sys.path.insert(0, REPO + "/studies/14nov")
_argv, sys.argv = sys.argv, sys.argv[:1]
import runcount as R  # noqa: E402  (loads the replay; its own main loop sees no orders)
sys.argv = _argv
y, ses, SC, p0, RUN, RUNS = R.y, R.ses, R.SC, R.p0, R.RUN, R.RUNS
IFACE = np.array(["Graz" if "Graz" in r.split("|")[1] else "BH" for r in RUN])
CELLS = {u: np.flatnonzero(np.char.add(ses.astype(str), IFACE) == u) for u in dict.fromkeys(np.char.add(ses.astype(str), IFACE))}
SESS = {u: np.flatnonzero(ses == u) for u in dict.fromkeys(ses)}
CNT = {u: np.bincount(y[r], minlength=3) for u, r in RUNS.items()}


def offline(P, groups):
    out = np.full_like(P, 1e-6)
    for r in groups.values():
        k = np.bincount(y[r], minlength=3); cols = np.repeat(np.arange(3), k)
        cost = -np.log(np.clip(P[r][:, cols], 1e-9, 1)); ri, ci = linear_sum_assignment(cost)
        out[r[ri], cols[ci]] = 1.0
    return out


def causal_exact(P, rk):
    out = P.copy()
    for u, r in RUNS.items():
        r = r[np.argsort(rk[r])]; cap = CNT[u]; alpha = {(0, 0, 0): 1.0}
        for i in r:
            L = np.clip(P[i], 1e-9, 1) ** R.TAU; L /= L.sum(); new, pred = {}, np.zeros(3)
            for k, a in alpha.items():
                tr = np.array([max(cap[c] - k[c], 0) + R.FLOOR for c in range(3)]); tr /= tr.sum()
                for c in range(3):
                    v = a * tr[c] * L[c]; pred[c] += v; kk = k[:c] + (k[c] + 1,) + k[c + 1:]; new[kk] = new.get(kk, 0.0) + v
            s = sum(new.values()); alpha = {k: v / s for k, v in new.items()}; out[i] = pred / pred.sum()
    return out


ARMS = ("b", "e", "c", "d", "f")
print("s248 M1 design ceiling: change in new-people BA against raw output, mean over 6 bases [P10 excluded in brackets]")
tot = {(a, arm): [] for a in ("P0r", "V6Fr") for arm in ARMS}; raw_lv = {"P0r": [], "V6Fr": []}
for order in sys.argv[1:]:
    rk = R.rank(order); z = np.load(f"{REPO}/runs/chainout/s215_{order}_off.npz"); bases = sorted({k.split("|")[0] for k in z.files})
    for a in ("P0r", "V6Fr"):
        res = {arm: [] for arm in ARMS}; raw = []
        for bn in bases:
            P = z[f"{bn}|{a}"].astype(np.float64); b0 = SC.person(P, p0); raw.append(b0)
            res["b"].append(SC.person(R.filt(P, rk), p0) - b0); res["e"].append(SC.person(causal_exact(P, rk), p0) - b0)
            res["c"].append(SC.person(offline(P, RUNS), p0) - b0); res["d"].append(SC.person(offline(P, CELLS), p0) - b0)
            res["f"].append(SC.person(offline(P, SESS), p0) - b0)
        raw_lv[a].append(np.mean(raw, 0).mean())
        line = []
        for arm in ARMS:
            d = np.mean(res[arm], 0); tot[a, arm].append(d); line.append(f"{arm} {d.mean():+.2f} [{np.delete(d, 9).mean():+.2f}]")
        print(f"  {order:8s} {a:4s} raw {np.mean(raw, 0).mean():.2f} | " + " | ".join(line), flush=True)
print("Mean over orders:")
for a in ("P0r", "V6Fr"):
    print(f"  {a:4s} raw {np.mean(raw_lv[a]):.2f} | " + " | ".join(
        f"{arm} {np.mean(tot[a, arm], 0).mean():+.2f} (min person {np.mean(tot[a, arm], 0).min():+.2f})" for arm in ARMS))
