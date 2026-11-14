"""EYE ORACLE (ledger s231): per-session best eye-view weight (labels: ceiling only) from shufcombo SAVE_OUT runs with
EYEW in {0, 0.5, 1, 2} (runs/chainout/s231_<order>_w<w>.npz, keys <base>|<arm>, recording layout). Reports, new people, v6f
(V6Fr): each fixed weight against w = 1, and the per-session oracle against w = 1, per order and scheme, with person intervals.
Usage: python studies/14nov/eyeoracle.py order [order ...]"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../.."); sys.path.insert(0, REPO + "/replay")
import env  # noqa: E402
A = env.A; SC = A.SC; p0 = "all_loso"; Z = A.L.Z; y, ses = Z[f"y|{p0}"].astype(int), Z[f"sess|{p0}"]; W = ("0", "0.5", "1", "2")
rng = np.random.default_rng(0)


def sacc(P, r):
    return np.mean([np.mean(P[r][y[r] == c].argmax(1) == c) for c in np.unique(y[r])])


print("EYE ORACLE (s231): change against w = 1, new people, V6Fr; oracle = per-session best w (labels, ceiling)")
allg = {}
for o in sys.argv[1:]:
    Zs = {w: np.load(f"{REPO}/runs/chainout/s231_{o}_w{w}.npz") for w in W}; bases = sorted({k.split("|")[0] for k in Zs["1"].files})
    res = {w: [] for w in W + ("oracle",)}
    for bn in bases:
        P = {w: Zs[w][f"{bn}|V6Fr"].astype(np.float64) for w in W}; base = SC.person(P["1"], p0)
        orc = P["1"].copy()
        for s in np.unique(ses):
            r = np.flatnonzero(ses == s); best = max(W, key=lambda w: sacc(P[w], r)); orc[r] = P[best][r]
        for w in W:
            res[w].append(SC.person(P[w], p0) - base)
        res["oracle"].append(SC.person(orc, p0) - base)
    m = {k: np.mean(v, 0) for k, v in res.items()}; allg[o] = m
    print(f"  {o:8s} " + " | ".join(f"w{k} {v.mean():+.2f}" for k, v in m.items() if k != "1"))
for sch in ("within", "full", "runs"):
    os_ = [o for o in allg if o.startswith(sch)]
    if os_:
        pm = np.mean([allg[o]["oracle"] for o in os_], 0); bs = [pm[rng.integers(0, 10, 10)].mean() for _ in range(2000)]
        print(f"  {sch:6s}: oracle {pm.mean():+.2f} [{np.percentile(bs, 10):+.2f}, {np.percentile(bs, 90):+.2f}] | " + " | ".join(f"w{w} {np.mean([allg[o][w].mean() for o in os_]):+.2f}" for w in W if w != "1"))
