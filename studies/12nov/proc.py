"""PROC (ledger s103c): labelled Procrustes oracle in core1's 256 components (new people). Usage: python studies/12nov/proc.py"""
import glob, os, sys
import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
REPO = os.path.abspath(__file__ + "/../../..")
sys.path.insert(0, REPO + "/replay")
import grid as G  # noqa: E402
p = "all_loso"; Z = G.Z; sess = Z[f"sess|{p}"]; y = Z[f"y|{p}"].astype(int); N = len(y); P0, P1 = np.zeros((N, 3)), np.zeros((N, 3))
zs = lambda X: (X - X.mean(0)) / (X.std(0) + 1e-6)  # noqa: E731  (whole-session standardisation; an oracle-side simplification)
for f in sorted(glob.glob(f"{REPO}/runs/replay_data/feats_exact/all_loso-P*.npz")):
    z = np.load(f, allow_pickle=True); Xtr, ytr, itr = z["pc_tr"], z["y_tr"].astype(int), z["idx_tr"].astype(int)
    Xs = np.zeros_like(Xtr)
    for s in np.unique(sess[itr]):
        m = sess[itr] == s; Xs[m] = zs(Xtr[m])
    clf = LDA(solver="lsqr", shrinkage="auto").fit(Xs, ytr); M = np.stack([Xs[ytr == k].mean(0) for k in range(3)])
    for k in [x for x in z.files if x.endswith("|pc")]:
        rows = z[k.replace("|pc", "|idx")].astype(int); X = zs(z[k]); P0[rows] = clf.predict_proba(X)
        Mt = np.stack([X[y[rows] == c].mean(0) for c in range(3)]); Rm, _ = orthogonal_procrustes(Mt, M)  # true labels: oracle
        P1[rows] = clf.predict_proba(X @ Rm)
a, b = G.scores(P0, p), G.scores(P1, p)
print(f"LDA in core1's 256 components: {a.mean():.1f} | with the labelled Procrustes rotation: {b.mean():.1f} ({b.mean() - a.mean():+.1f}) | " + " ".join(f"{x:+.0f}" for x in b - a))
