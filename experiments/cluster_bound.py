"""Bound for label-free per-session adaptation by clustering (R8).

For every session (stream): v2 features centred on that session (non-causal,
an upper bound), standardised and reduced by a PCA fitted on the session
alone; k-means / GMM with 3 clusters; accuracy of the clusters mapped to
classes (a) by the true labels (oracle Hungarian) and (b) by the r6nf
all_loso posteriors (label-free: cluster c -> class maximising the summed
posterior, one-to-one). Compared with r6nf's own accuracy on that session.
Usage: python experiments/cluster_bound.py [DIMS ...]
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "scripts")]
import protocol  # noqa: E402
import run_cv  # noqa: E402
from ensemble import rebalance  # noqa: E402
from mix_probs import member  # noqa: E402
from models import TangentSpace  # noqa: E402

d = protocol.load()
DIRS = "probs,probs_cloud,probs_r5,probs_local,probs_r7".split(",")
fam = ["v2", "shallow_ea", "shallow_raw", "fbcnet_ea", "fbcnet_raw"]
five = {str(i) for i in range(5)}
F = [member(f, None if f == "v2" else five, "all_loso", DIRS)[:2] for f in fam]
pidx = F[0][1]
P = sum(f[0] for f in F) / 5
post = np.zeros((len(d["y"]), 3))
st = d["stream"][pidx]
for s in dict.fromkeys(st):
    post[pidx[st == s]] = rebalance(P[st == s], 1, 5, 30)

ts = TangentSpace(d, **run_cv.V2)
ts.fit(np.arange(len(d["y"])))  # global references only; the classifier is unused


def mapped(lab, score):
    """One-to-one cluster -> class map maximising ``score`` (3 x 3)."""
    r, c = linear_sum_assignment(-score)
    m = dict(zip(r, c))
    return np.array([m[x] for x in lab])


def bacc(y, p):
    return np.mean([np.mean(p[y == c] == c) for c in range(3)])


dims = [int(x) for x in sys.argv[1:]] or [5, 10, 20]
rows = []
for s in dict.fromkeys(d["stream"]):
    idx = np.flatnonzero(d["stream"] == s)
    X = ts._session_feats(idx, causal=False)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    y, g = d["y"][idx], post[idx]
    row = {"stream": s.split("/")[-1], "global": bacc(y, g.argmax(1))}
    for k in dims:
        Z = PCA(k, random_state=0).fit_transform(X)
        for name, lab in (("km", KMeans(3, n_init=20, random_state=0).fit_predict(Z)),
                          ("gmm", GaussianMixture(3, covariance_type="diag", n_init=5,
                                                  random_state=0).fit_predict(Z))):
            onehot = np.eye(3)[lab]
            row[f"{name}{k}_oracle"] = bacc(y, mapped(lab, onehot.T @ np.eye(3)[y]))
            row[f"{name}{k}_free"] = bacc(y, mapped(lab, onehot.T @ g))
    rows.append(row)
    print(row["stream"], " ".join(f"{k}={v:.2f}" for k, v in row.items() if k != "stream"),
          flush=True)
keys = [k for k in rows[0] if k != "stream"]
print("MEAN", " ".join(f"{k}={np.mean([r[k] for r in rows]):.3f}" for k in keys))
