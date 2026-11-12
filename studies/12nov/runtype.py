"""RUNTYPE (ledger s106): calibration (Acquisition*) and feedback (OnlineRaw*) runs as separate session domains, oracle
labels; grid chain, new people, six bases. Usage: python studies/12nov/runtype.py"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
sys.path.insert(0, REPO + "/replay")
import env  # noqa: E402
import grid as G  # noqa: E402
p = "all_loso"; Z = G.Z; sess = Z[f"sess|{p}"]; TK = np.load(REPO + "/data/proteus_windows.npz", allow_pickle=True)["task"].astype(str)
assert (Z[f"idx|{p}"] == np.arange(len(sess))).all(); ST = np.array([f"{s}|{'acq' if t.startswith('Acq') else 'onl'}" for s, t in zip(sess, TK)]); rng = np.random.default_rng(0)


def chain(base, rb_by, st_by):
    Ps, ws = Z[f"{base}|Ps|{p}"], Z[f"{base}|ws|{p}"].copy(); ws[0], ws[-1] = 2.0, 5.0; mix = sum(w / ws.sum() * P for w, P in zip(ws, Ps))
    b = G.rebal(mix, rb_by, 1.15); e = G.SE.apply(Z[f"GE|{p}"], b, st_by, lam=0.5, n0=60, m0=20, shrink=0.7)
    L, tt = G.tsem_lik(e, st_by, Z[f"pc128|{p}"]); return G.scores(G.apply_lam(e, L, tt, 0.15), p)


def boot(d):
    b = [d[rng.integers(0, 10, 10)].mean() for _ in range(4000)]; return f"{d.mean():+.2f} [{np.percentile(b, 10):+.2f}, {np.percentile(b, 90):+.2f}]"


D1, D2 = [], []
for bn in env.BASES:
    ref = chain(bn, sess, sess); d1 = chain(bn, ST, sess) - ref; d2 = chain(bn, sess, ST) - ref; D1.append(d1); D2.append(d2)
    print(f"{bn}: rebalancing per run type {d1.mean():+.2f} | SessionEM + TSEM per run type {d2.mean():+.2f}", flush=True)
print(f"MEAN six bases: rebalancing per run type {boot(np.mean(D1, 0))} | stages per run type {boot(np.mean(D2, 0))}")
