"""WARMUP (ledger s90): session stages run twice over each session; windows scored from the second pass (hindsight bound).
Usage: python studies/12nov/warmup.py"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
sys.path.insert(0, REPO + "/replay")
import grid as G  # noqa: E402
p = "all_loso"; Z = G.Z; sess = Z[f"sess|{p}"]; rng = np.random.default_rng(0)
order = np.concatenate([np.concatenate([np.flatnonzero(sess == s)] * 2) for s in np.unique(sess)])
second = np.concatenate([np.r_[np.zeros(np.sum(sess == s), bool), np.ones(np.sum(sess == s), bool)] for s in np.unique(sess)])
pos = np.zeros(len(sess))
for s in np.unique(sess):
    r = np.flatnonzero(sess == s); pos[r] = np.arange(len(r)) / len(r)


def chain(base, rows, sid):
    Ps, ws = Z[f"{base}|Ps|{p}"][:, rows], Z[f"{base}|ws|{p}"].copy(); ws[0], ws[-1] = 2.0, 5.0
    mix = sum(w / ws.sum() * P for w, P in zip(ws, Ps)); b = G.rebal(mix, sid, 1.15)
    e = G.SE.apply(Z[f"GE|{p}"][rows], b, sid, lam=0.5, n0=60, m0=20, shrink=0.7); L, tt = G.tsem_lik(e, sid, Z[f"pc128|{p}"][rows]); return G.apply_lam(e, L, tt, 0.15)


y = Z[f"y|{p}"].astype(int); D = []
for base in ("B2_pool", "B2_s345", "B2_s012", "B2_s0", "B2_s1", "B2_s2"):
    one = chain(base, np.arange(len(sess)), sess); two = np.zeros_like(one); two[order[second]] = chain(base, order, sess[order])[second]
    a, b = G.scores(one, p), G.scores(two, p); d = b - a; D.append(d)
    q = lambda P, lo, hi: 100 * np.mean(P[(pos >= lo) & (pos < hi)].argmax(1) == y[(pos >= lo) & (pos < hi)])  # noqa: E731
    print(f"{base}: causal {a.mean():.2f} -> hindsight {b.mean():.2f} ({d.mean():+.2f}) | window accuracy first fifth {q(one, 0, .2):.1f} -> {q(two, 0, .2):.1f}, last fifth {q(one, .8, 1.01):.1f} -> {q(two, .8, 1.01):.1f} | " + " ".join(f"{x:+.1f}" for x in d), flush=True)
m = np.mean(D, 0); bt = np.percentile([m[rng.integers(0, 10, 10)].mean() for _ in range(4000)], [10, 90])
print(f"MEAN of six bases: {m.mean():+.2f} [{bt[0]:+.2f}, {bt[1]:+.2f}]")
