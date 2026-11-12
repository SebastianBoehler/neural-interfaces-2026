"""EARLY (ledger s113, explore): a window-local system for the start of a session, needing no session statistics:
relative log band power per channel (each band's power over the window's total) and the tangent space at the training
people's grand mean (no recentring); logistic regression, leave one person out. Scored on the first 15 and 30 windows
of each new session against the routed chain, and fused there with a weight that fades out by window 30.
Usage: python studies/12nov/early.py [smoke]"""
import os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
REPO = os.path.abspath(__file__ + "/../../..")
sys.path[:0] = [REPO + "/replay", REPO + "/studies/12nov"]
import env  # noqa: E402
import grid as G  # noqa: E402
import ownspace as O  # noqa: E402
A = env.A; p = "all_loso"; Z = A.L.Z; sess = Z[f"sess|{p}"]; y = O.y.astype(int); N = len(y); rng = np.random.default_rng(0)
SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"; PEOPLE = sorted(set(O.PER))[:2] if SMOKE else sorted(set(O.PER))
pos = np.zeros(N, int)
for s in np.unique(sess):
    r = np.flatnonzero(sess == s); pos[r] = np.arange(len(r))
bp = np.log(np.stack([np.einsum("ncc->nc", C) for C in O.CV], 2)); rel = (bp - np.log(np.exp(bp).sum(2, keepdims=True))).reshape(N, -1)
P = np.full((N, 3), 1 / 3)
for q in PEOPLE:
    te = O.PER == q; Cw = [O._isqrt(C[~te].mean(0))[None] @ C @ O._isqrt(C[~te].mean(0))[None] for C in O.CV]  # grand-mean reference
    X = np.c_[rel, O.tang(Cw)]; P[te] = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=1000)).fit(X[~te], y[~te]).predict_proba(X[te]); print("fold", q, flush=True)
C0 = A.system("B2_pool", A.FAMS)["out"][p]; m = np.isin(O.PER, PEOPLE)
for lo, hi in ((0, 15), (0, 30), (30, 10 ** 6)):
    k = m & (pos >= lo) & (pos < hi); print(f"windows {lo}-{hi}: early system {100 * np.mean(P[k].argmax(1) == y[k]):.1f}% | chain {100 * np.mean(C0[k].argmax(1) == y[k]):.1f}%")
for bn in ("B2_pool",) if SMOKE else env.BASES:
    C = A.system(bn, A.FAMS)["out"][p]; ref = G.scores(C, p)
    for w0 in (0.5, 1.0):
        w = (w0 * np.clip(1 - pos / 30, 0, 1))[:, None]; L = np.log(np.clip(C, 1e-9, 1)) + w * np.log(np.clip(P, 1e-3, 1)); e = np.exp(L - L.max(1, keepdims=True))
        d = G.scores(e / e.sum(1, keepdims=True), p) - ref; print(f"  {bn}: early system fused (weight {w0} fading by window 30): {d.mean():+.2f} people >= 0 {(d >= 0).sum()}/10", flush=True)
