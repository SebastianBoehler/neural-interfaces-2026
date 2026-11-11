"""CLUS diagnostic (exploratory): do a new person's windows cluster into the true classes without labels?
Per session and feature space: (a) best-permutation accuracy of unsupervised 3-cluster models at session end
(transductive ceiling of 'cluster first, name later'); (b) the same clusters named by the chain's summed posteriors;
(c) within-session 5-fold supervised accuracy (what labels would give); against the chain's own accuracy."""
import itertools, sys
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold, cross_val_predict
REPO = __import__("os").path.abspath(__file__ + "/../../../..")
sys.path.insert(0, REPO + "/replay")
import env
A = env.A; L = A.L; p = "all_loso"
y, ss, per, pc = L.Z[f"y|{p}"], L.Z[f"sess|{p}"], L.Z[f"person|{p}"], L.Z[f"pc128|{p}"]
Ps = L.Z[f"B2_pool|Ps|{p}"]; chain = A.system("B2_pool", A.FAMS)["out"][p]
logp = np.concatenate([np.log(np.clip(m, 1e-4, 1)) for m in Ps], 1)  # 33 dims: members' log posteriors
PERMS = list(itertools.permutations(range(3)))


def best_perm(c, yy):
    return max(np.mean(np.array(pm)[c] == yy) for pm in PERMS)


def named(c, yy, P):
    pm = max(PERMS, key=lambda pm: sum(P[c == k, pm[k]].sum() for k in range(3))); return np.mean(np.array(pm)[c] == yy)


def lda_space(r, d=2):
    """LDA directions fitted on the other people (session-centred pc128), applied to the session."""
    o = per != per[r[0]]; Xo = pc[o].copy()
    for s in np.unique(ss[o]):
        m = ss[o] == s; Xo[m] -= Xo[m].mean(0)
    l = LDA(solver="eigen", shrinkage="auto").fit(Xo, y[o]); return (pc[r] - pc[r].mean(0)) @ l.scalings_[:, :d]


rows = []
for s in np.unique(ss):
    r = np.flatnonzero(ss == s); yy = y[r]; X = pc[r] - pc[r].mean(0)
    spaces = {"pc8": X[:, :8], "pc32": X[:, :32], "pc128": X, "lda2": lda_space(r, 2), "logp33": logp[r] - logp[r].mean(0)}
    res = {"chain": np.mean(chain[r].argmax(1) == yy)}
    for nm, Z in spaces.items():
        Z = Z / Z.std(0).mean()
        km = KMeans(3, n_init=20, random_state=0).fit(Z).labels_
        cov = "tied" if Z.shape[1] <= 33 else "spherical"
        gm = GaussianMixture(3, covariance_type=cov, n_init=10, random_state=0, reg_covar=1e-3).fit(Z).predict(Z)
        gs = GaussianMixture(3, covariance_type=cov, random_state=0, reg_covar=1e-3,
                             means_init=np.stack([Z[chain[r].argmax(1) == k].mean(0) for k in range(3)])).fit(Z).predict(Z)
        cv = cross_val_predict(LDA(solver="lsqr", shrinkage="auto"), Z, yy, cv=StratifiedKFold(5, shuffle=True, random_state=0))
        res[nm] = (best_perm(km, yy), best_perm(gm, yy), named(gm, yy, chain[r]), np.mean(gs == yy), np.mean(cv == yy))
    rows.append((per[r[0]], s, len(r), res))
    print(per[r[0]], s, len(r), f"chain {res['chain']:.3f} | " + " | ".join(f"{nm} km {v[0]:.2f} gmm {v[1]:.2f} named {v[2]:.2f} seeded {v[3]:.2f} sup {v[4]:.2f}" for nm, v in res.items() if nm != "chain"), flush=True)
print("\nmean over sessions: chain", np.mean([r[3]["chain"] for r in rows]).round(3))
for nm in ("pc8", "pc32", "pc128", "lda2", "logp33"):
    v = np.mean([r[3][nm] for r in rows], 0)
    print(f"{nm:7s} k-means best-perm {v[0]:.3f} | GMM best-perm {v[1]:.3f} | GMM chain-named {v[2]:.3f} | GMM chain-seeded {v[3]:.3f} | supervised 5-fold {v[4]:.3f}")
