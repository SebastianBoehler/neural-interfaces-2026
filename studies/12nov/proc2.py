"""PROC2 (ledger s104): causal Procrustes alignment of a new session onto the population class means (true-label oracle
and label-free from the chain's confident windows), population LDA in core1's 256 components. Usage: python studies/12nov/proc2.py"""
import glob, os, sys
import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
REPO = os.path.abspath(__file__ + "/../../..")
sys.path.insert(0, REPO + "/replay")
import env  # noqa: E402
import grid as G  # noqa: E402
A = env.A; p = "all_loso"; Z = G.Z; sess = Z[f"sess|{p}"]; y = Z[f"y|{p}"].astype(int); N = len(y); rng = np.random.default_rng(0)
CH = A.system("B2_pool", A.FAMS)["out"][p]; PO, PL = np.full((N, 3), 1 / 3), np.full((N, 3), 1 / 3)


def czs(X):  # causal standardisation: window t by windows 0..t
    n = np.arange(1, len(X) + 1)[:, None]; m = np.cumsum(X, 0) / n; v = np.cumsum(X ** 2, 0) / n - m ** 2; return (X - m) / np.sqrt(np.maximum(v, 1e-6))


for f in sorted(glob.glob(f"{REPO}/runs/replay_data/feats_exact/all_loso-P*.npz")):
    z = np.load(f, allow_pickle=True); Xtr, ytr, itr = z["pc_tr"], z["y_tr"].astype(int), z["idx_tr"].astype(int); Xs = np.zeros_like(Xtr)
    for s in np.unique(sess[itr]):
        m = sess[itr] == s; Xs[m] = (Xtr[m] - Xtr[m].mean(0)) / (Xtr[m].std(0) + 1e-6)
    clf = LDA(solver="lsqr", shrinkage="auto").fit(Xs, ytr); M = np.stack([Xs[ytr == k].mean(0) for k in range(3)])
    for k in [x for x in z.files if x.endswith("|pc")]:
        rows = z[k.replace("|pc", "|idx")].astype(int); X = czs(z[k]); lab, cf = CH[rows].argmax(1), CH[rows].max(1)
        for t in range(15, len(rows)):
            for out, src in ((PO, y[rows]), (PL, lab)):
                if src is lab:  # TSEM's selection: the q most confident earlier windows per predicted class
                    q = max(2, int(0.7 * t / 3)); sel = [np.flatnonzero(lab[:t] == c)[np.argsort(-cf[:t][lab[:t] == c])[:q]] for c in range(3)]
                else:
                    sel = [np.flatnonzero(src[:t] == c) for c in range(3)]
                if min(len(x) for x in sel) < 2:
                    continue
                Mt = np.stack([X[x].mean(0) for x in sel]); R, _ = orthogonal_procrustes(Mt, M); out[rows[t]] = clf.predict_proba(X[t:t + 1] @ R)[0]
print(f"causal true-label means, alone: {G.scores(PO, p).mean():.1f} | label-free means, alone: {G.scores(PL, p).mean():.1f}", flush=True)
for bn in env.BASES:
    C = A.system(bn, A.FAMS)["out"][p]; ref = G.scores(C, p)
    for w in (0.25, 0.5):
        L = np.log(np.clip(C, 1e-9, 1)) + w * np.log(np.clip(PL, 1e-3, 1)); e = np.exp(L - L.max(1, keepdims=True)); d = G.scores(e / e.sum(1, keepdims=True), p) - ref
        bt = np.percentile([d[rng.integers(0, 10, 10)].mean() for _ in range(4000)], [10, 90]); print(f"  {bn}: chain x label-free Procrustes^{w}: {d.mean():+.2f} [{bt[0]:+.2f}, {bt[1]:+.2f}] people >= 0 {(d >= 0).sum()}/10", flush=True)
