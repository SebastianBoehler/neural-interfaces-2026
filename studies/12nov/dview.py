"""DVIEW (ledger s74): discriminative views from core1's cached 256 components as a second TSEM after the routed chain.
Usage: python studies/12nov/dview.py"""
import glob, os, sys
import numpy as np
from sklearn.cross_decomposition import PLSRegression
REPO = os.path.abspath(__file__ + "/../../..")
sys.path.insert(0, REPO + "/replay")
import env  # noqa: E402
import grid as G  # noqa: E402
A = env.A; p = "all_loso"; Z = A.L.Z; sess = Z[f"sess|{p}"]; per = Z[f"person|{p}"]; N = len(sess); rng = np.random.default_rng(0)
assert (Z[f"idx|{p}"] == np.arange(N)).all()


def lda_dirs(X, y, shrink=0.5, k=2):
    mu = X.mean(0); Sw = sum(np.cov((X[y == c] - X[y == c].mean(0)).T) for c in range(3)); Sw = (1 - shrink) * Sw + shrink * np.eye(len(Sw)) * np.trace(Sw) / len(Sw)
    Sb = sum(np.outer(X[y == c].mean(0) - mu, X[y == c].mean(0) - mu) for c in range(3)); w, V = np.linalg.eig(np.linalg.solve(Sw, Sb)); return np.real(V[:, np.argsort(-np.real(w))[:k]])


def orth(W):
    return np.linalg.qr(W)[0]


VIEWS = {k: np.zeros((N, d)) for k, d in (("lda2", 2), ("pls16", 16), ("perperson18", 18), ("random18", 18))}
for f in sorted(glob.glob(f"{REPO}/runs/replay_data/feats_exact/all_loso-P*.npz")):
    z = np.load(f, allow_pickle=True); Xtr, ytr, itr = z["pc_tr"], z["y_tr"].astype(int), z["idx_tr"].astype(int)
    Xc = Xtr.copy()
    for s in np.unique(sess[itr]):
        m = sess[itr] == s; Xc[m] -= Xc[m].mean(0)
    sd = Xc.std(0); Xc = Xc / sd
    W = {"lda2": lda_dirs(Xc, ytr), "pls16": PLSRegression(16, scale=False).fit(Xc, np.eye(3)[ytr]).x_rotations_,
         "perperson18": orth(np.hstack([lda_dirs(Xc[per[itr] == q], ytr[per[itr] == q]) for q in np.unique(per[itr])])), "random18": orth(rng.standard_normal((256, 18)))}
    for k in [x for x in z.files if x.endswith("|pc")]:
        rows = z[k.replace("|pc", "|idx")].astype(int); Xt = z[k] / sd
        for v, Wv in W.items():
            VIEWS[v][rows] = Xt @ Wv
assert all(np.abs(V).sum(1).min() > 0 for V in VIEWS.values()), "a test row was never filled"


def boot(d):
    b = [d[rng.integers(0, 10, 10)].mean() for _ in range(4000)]; return f"{d.mean():+.2f} [{np.percentile(b, 10):+.2f}, {np.percentile(b, 90):+.2f}]"


for bn in env.EXPL:
    base = A.system(bn, A.FAMS)["out"][p]; ref = G.scores(base, p); print(f"{bn}: routed chain {ref.mean():.2f}", flush=True)
    for v, X in VIEWS.items():
        L, tt = G.tsem_lik(base, sess, X)
        for lam in (0.10, 0.15):
            d = G.scores(G.apply_lam(base, L, tt, lam), p) - ref; print(f"  {v:12s} lambda {lam}: {boot(d)} | P04 {d[3]:+.1f} | people >= 0: {(d >= 0).sum()}/10 | " + " ".join(f"{x:+.1f}" for x in d), flush=True)
