"""BANDMAP (ledger s77): per band and family, accuracy from own other sessions (P04, P10) and from the pooled other nine.
Usage: python studies/12nov/bandmap.py"""
import os, sys
import numpy as np
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.abspath(__file__ + "/.."))
import ownspace as O  # noqa: E402
PEOPLE = sorted(set(O.PER)); NB = len(O.BANDS)
sess = lambda p: [np.flatnonzero((O.PER == p) & (O.SES == s)) for s in sorted(set(O.SES[O.PER == p]))]  # noqa: E731


def fit(train):
    cw, ms = zip(*[O.recentred(r) for r in train]); yt = np.concatenate([O.y[r] for r in train]); M = {}
    prior = [np.mean([m[b] for m in ms], 0) for b in range(NB)]
    for b in range(NB):
        C = np.concatenate([c[b] for c in cw]); Ft = O.tang([C]); M["T", b] = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=500)).fit(Ft, yt)
        W = np.concatenate([eigh(C[yt == a].mean(0), C[yt == a].mean(0) + C[yt == c].mean(0))[1][:, [0, 1, 2, -3, -2, -1]] for a, c in ((0, 1), (0, 2), (1, 2))], 1)
        M["C", b] = (W, LDA(solver="lsqr", shrinkage=0.5).fit(np.log(np.einsum("cf,ncd,df->nf", W, C, W)), yt))
    S = np.concatenate([(O.SL[r] - O.SL[r].mean(0)) / (O.SL[r].std(0) + 1e-9) for r in train]); M["S"] = LDA(solver="lsqr", shrinkage=0.7).fit(S, yt)
    return M, prior


def score(M, prior, r):
    Cw, _ = O.recentred(r, prior); out = {}
    for b in range(NB):
        out["T", b] = O.ba(r, M["T", b].predict_proba(O.tang([Cw[b]]))); W, m = M["C", b]; out["C", b] = O.ba(r, m.predict_proba(np.log(np.einsum("cf,ncd,df->nf", W, Cw[b], W))))
    out["S"] = O.ba(r, M["S"].predict_proba(O.causal_z(O.SL[r]))); return out


keys = [(f, b) for f in "TC" for b in range(NB)] + ["S"]; name = lambda k: k if k == "S" else f"{k[0]}{O.BANDS[k[1]][0]}-{O.BANDS[k[1]][1]}"  # noqa: E731
X = {}
for t in PEOPLE:
    M, pr = fit([r for q in PEOPLE if q != t for r in sess(q)]); res = [score(M, pr, r) for r in sess(t)]; X["others", t] = {k: np.mean([o[k] for o in res]) for k in keys}
    if len(sess(t)) > 1:
        own = []
        for r in sess(t):
            M2, pr2 = fit([q for q in sess(t) if q[0] != r[0]]); own.append(score(M2, pr2, r))
        X["own", t] = {k: np.mean([o[k] for o in own]) for k in keys}
    print("done", t, flush=True)
print("feature    " + " ".join(f"{q[1:]:>4s}" for q in PEOPLE) + " | others-mean | P04 own | P10 own")
for k in keys:
    v = [X["others", q][k] for q in PEOPLE]
    print(f"{name(k):10s} " + " ".join(f"{x:4.0f}" for x in v) + f" | {np.mean([x for q, x in zip(PEOPLE, v) if q != 'P04']):5.1f} | {X['own', 'P04'][k]:5.1f} | {X['own', 'P10'][k]:5.1f}")
