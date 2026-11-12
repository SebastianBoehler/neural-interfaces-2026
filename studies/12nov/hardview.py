"""HARDVIEW (ledger s86): population classifiers on other sensor views, measured and fused on the chain's hard windows.
Usage: python studies/12nov/hardview.py"""
import os, sys
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
REPO = os.path.abspath(__file__ + "/../../..")
sys.path[:0] = [REPO + "/replay", REPO + "/studies/12nov"]
import env  # noqa: E402
import grid as G  # noqa: E402
import ownspace as O  # noqa: E402
A = env.A; Z = A.L.Z; p = "all_loso"; sess = Z[f"sess|{p}"]; y = O.y.astype(int); N = len(y); rng = np.random.default_rng(0)
assert (Z[f"idx|{p}"] == np.arange(N)).all() and (Z[f"y|{p}"].astype(int) == y).all()
PEOPLE = sorted(set(O.PER))
bp = np.log(np.stack([np.einsum("ncc->nc", C) for C in O.CV], 2))  # (n, 41, 9) per-channel band power
B = sosfiltfilt(butter(4, [4, 45], btype="band", fs=120, output="sos"), O.Xc, axis=-1)
VIEWS = {"gaze": Z[f"GE|{p}"], "muscle": bp[:, :, 5:7].reshape(N, -1), "bandpow": bp.reshape(N, -1),
         "tprofile": np.log(np.stack([(B[:, :, i * 60:(i + 1) * 60] ** 2).mean(-1) for i in range(8)], 2)).reshape(N, -1), "slow": O.SL}
P = {}
for v, X in VIEWS.items():
    Xz = np.zeros_like(X, dtype=np.float64)
    for s in np.unique(sess):
        r = np.flatnonzero(sess == s); Xz[r] = O.causal_z(X[r].astype(np.float64))
    out = np.zeros((N, 3))
    for q in PEOPLE:
        te = O.PER == q; out[te] = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=1000)).fit(Xz[~te], y[~te]).predict_proba(Xz[te])
    P[v] = out; print(f"view {v} ({X.shape[1]} features): balanced accuracy, new people {G.scores(out, p).mean():.1f}", flush=True)


def hard_mask(C):
    m = np.zeros(N, bool); cf = C.max(1)
    for s in np.unique(sess):
        r = np.flatnonzero(sess == s); m[r[cf[r] <= np.quantile(cf[r], 1 / 3)]] = True
    return m


def boot(d):
    b = [d[rng.integers(0, 10, 10)].mean() for _ in range(4000)]; return f"{d.mean():+.2f} [{np.percentile(b, 10):+.2f}, {np.percentile(b, 90):+.2f}]"


ARMS = [(v, w, c) for v in VIEWS for w in (0.25, 0.5, 1.0) for c in (0.5, 0.6, 1.01)]; D = {}
for bn in env.BASES:
    C = A.system(bn, A.FAMS)["out"][p]; ref = G.scores(C, p)
    if bn == "B2_pool":
        h = hard_mask(C); wr = h & (C.argmax(1) != y)
        print(f"hard windows: {h.mean():.2f} of all; chain accuracy there {100 * np.mean(C[h].argmax(1) == y[h]):.1f}% (elsewhere {100 * np.mean(C[~h].argmax(1) == y[~h]):.1f}%)")
        for v in VIEWS:
            pr = P[v].argmax(1); print(f"  {v:9s} on hard {100 * np.mean(pr[h] == y[h]):.1f}% | on hard where the chain is wrong: true class {100 * np.mean(pr[wr] == y[wr]):.1f}%, chain's class {100 * np.mean(pr[wr] == C[wr].argmax(1)):.1f}%", flush=True)
    for v, w, c in ARMS:
        g = (C.max(1) < c).astype(float)[:, None]; L = np.log(np.clip(C, 1e-9, 1)) + w * g * np.log(np.clip(P[v], 1e-3, 1)); e = np.exp(L - L.max(1, keepdims=True))
        D[bn, v, w, c] = G.scores(e / e.sum(1, keepdims=True), p) - ref
best = max(ARMS, key=lambda a: np.mean([D[b, *a].mean() for b in env.EXPL]))
print("top arms on the exploration bases: " + " | ".join(f"{a}: {np.mean([D[b, *a].mean() for b in env.EXPL]):+.2f}" for a in sorted(ARMS, key=lambda a: -np.mean([D[b, *a].mean() for b in env.EXPL]))[:5]))
fresh = np.mean([D[b, *best] for b in env.FRESH], 0)
print(f"chosen arm {best}: exploration {np.mean([D[b, *best].mean() for b in env.EXPL]):+.2f} | FRESH four bases {boot(fresh)} | people >= 0 {(fresh >= 0).sum()}/10 | " + " ".join(f"{x:+.1f}" for x in fresh))
for v in VIEWS:
    a = max([x for x in ARMS if x[0] == v], key=lambda a: np.mean([D[b, *a].mean() for b in env.EXPL])); f = np.mean([D[b, *a] for b in env.FRESH], 0)
    print(f"  best arm per view {a}: exploration {np.mean([D[b, *a].mean() for b in env.EXPL]):+.2f} | fresh {boot(f)}")
