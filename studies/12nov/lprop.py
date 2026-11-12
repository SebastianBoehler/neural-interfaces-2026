"""LPROP (ledger s89): label spreading in the session's own space from the chain's easy windows to its hard ones.
Non-causal ceiling. Usage: python studies/12nov/lprop.py"""
import os, sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.semi_supervised import LabelSpreading
REPO = os.path.abspath(__file__ + "/../../..")
sys.path[:0] = [REPO + "/replay", REPO + "/studies/12nov"]
import env  # noqa: E402
import ownspace as O  # noqa: E402
A = env.A; Z = A.L.Z; p = "all_loso"; sess = Z[f"sess|{p}"]; y = O.y.astype(int)
C = A.system("B2_pool", A.FAMS)["out"][p]; cf, pc = C.max(1), C.argmax(1); res = {}
for s in np.unique(sess):
    r = np.flatnonzero(sess == s); Cw, _ = O.recentred(r); F = O.tang(Cw); F = (F - F.mean(0)) / (F.std(0) + 1e-9)
    X = PCA(30, random_state=0).fit_transform(F); h = cf[r] <= np.quantile(cf[r], 1 / 3)
    for k in (7, 15):
        for tag, lab in (("true", y[r]), ("chain", pc[r])):
            seed = np.where(h, -1, lab); m = LabelSpreading(kernel="knn", n_neighbors=k, alpha=0.5, max_iter=200).fit(X, seed)
            res.setdefault((k, tag), []).append((m.transduction_[h] == y[r][h], pc[r][h] == y[r][h], m.transduction_[h], y[r][h], pc[r][h]))
for (k, tag), L in sorted(res.items()):
    acc = np.concatenate([a for a, *_ in L]); ch = np.concatenate([b for _, b, *_ in L]); t = np.concatenate([x[2] for x in L]); yy = np.concatenate([x[3] for x in L]); cc = np.concatenate([x[4] for x in L])
    w = cc != yy; print(f"k {k:2d} seeds {tag:5s}: hard-window accuracy {100 * acc.mean():.1f}% (chain {100 * ch.mean():.1f}%) | where the chain is wrong: true class {100 * np.mean(t[w] == yy[w]):.1f}%, chain's class {100 * np.mean(t[w] == cc[w]):.1f}%")
