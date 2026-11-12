"""FAMSEL (ledger s117, explore): per-session member-family weights. 17 weightings of the mix (shipped; each family
dropped; each family doubled) run through the grid chain; per new session the weighting is chosen (oracle: true
accuracy; label-free: agreement with the majority of the 17 outputs). Usage: python studies/12nov/famsel.py [smoke]"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
sys.path.insert(0, REPO + "/replay")
import env  # noqa: E402
import grid as G  # noqa: E402
SMOKE = "smoke" in sys.argv; p = "all_loso"; Z = G.Z; sess = Z[f"sess|{p}"]; y = Z[f"y|{p}"].astype(int); cell = Z[f"cell|{p}"]; rng = np.random.default_rng(0)
FAM = {"core": [0], "tf": [1, 2], "shallow": [3, 4], "sbnfb": [5, 6, 9], "sbnsinc": [7], "sbnfbl": [8], "teacher": [10]}


def chain(base, scale):
    Ps, ws = Z[f"{base}|Ps|{p}"], Z[f"{base}|ws|{p}"].copy(); ws[0], ws[-1] = 2.0, 5.0; ws = ws * scale
    mix = sum(w / ws.sum() * P for w, P in zip(ws, Ps)); b = G.rebal(mix, sess, 1.15)
    e = G.SE.apply(Z[f"GE|{p}"], b, sess, lam=0.5, n0=60, m0=20, shrink=0.7); L, tt = G.tsem_lik(e, sess, Z[f"pc128|{p}"]); return G.apply_lam(e, L, tt, 0.15)


W = [("shipped", np.ones(11))]
for f, ix in FAM.items():
    for tag, v in (("drop", 0.0), ("x2", 2.0)):
        sc = np.ones(11); sc[ix] = v; W.append((f"{tag} {f}", sc))
W = W[:3] if SMOKE else W
for base in ("B2_pool",) if SMOKE else env.BASES:
    O = {n: chain(base, sc) for n, sc in W}; ref = G.scores(O["shipped"], p); pick = {"oracle": O["shipped"].copy(), "agree": O["shipped"].copy()}
    maj = np.stack([v.argmax(1) for v in O.values()]); maj = np.array([np.bincount(c, minlength=3).argmax() for c in maj.T])
    for s in np.unique(sess):
        r = np.flatnonzero(sess == s)
        ba = {n: np.mean([np.mean(v[r][(cell[r] == c) & (y[r] == k)].argmax(1) == k) for c in np.unique(cell[r]) for k in range(3) if ((cell[r] == c) & (y[r] == k)).any()]) for n, v in O.items()}
        pick["oracle"][r] = O[max(O, key=lambda n: ba[n])][r]; pick["agree"][r] = O[max(O, key=lambda n: np.mean(O[n][r].argmax(1) == maj[r]))][r]
    print(f"{base}: " + " | ".join(f"{n} {G.scores(v, p).mean() - ref.mean():+.2f}" for n, v in O.items()), flush=True)
    for k, P in pick.items():
        d = G.scores(P, p) - ref; bt = np.percentile([d[rng.integers(0, 10, 10)].mean() for _ in range(4000)], [10, 90]); print(f"  per-session pick by {k}: {d.mean():+.2f} [{bt[0]:+.2f}, {bt[1]:+.2f}]", flush=True)
