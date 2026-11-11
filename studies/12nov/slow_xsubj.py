"""Slow-potential / gaze view: within-session supervised vs cross-subject (LOSO), per interface and person.
Question: is the gaze-to-class mapping shared across people (a fixed screen layout), or person-specific?"""
import sys, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score as bas
z = np.load("data/proteus_windows.npz", allow_pickle=True)
X, y, sub, ses, itf = z["X"], z["y"], np.array([s[-3:] for s in z["subject"]]), z["session"], z["interface"]
sid = np.array([f"{a}-{b}" for a, b in zip(sub, ses)])
def feats(X):
    seg = X.reshape(len(X), 41, 4, 120).mean(-1)            # four 1-s means per channel
    f = np.concatenate([seg.reshape(len(X), -1), X.mean(-1)], 1)
    return f
F = feats(X)
# per-session standardisation (label-free, what a causal session normaliser converges to)
Fz = F.copy()
for s in np.unique(sid):
    m = sid == s; Fz[m] = (F[m] - F[m].mean(0)) / (F[m].std(0) + 1e-9)
clf = lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
print("cell            n  within  loso")
rows = []
for itn in ("Graz", "BrainHero"):
    for s in np.unique(sid):
        m = (sid == s) & (itf == itn)
        pw = cross_val_predict(clf(0.01), Fz[m], y[m], cv=StratifiedKFold(5, shuffle=True, random_state=0))
        tr = (sub != s[:3]) & (itf == itn)
        pl = clf(0.01).fit(Fz[tr], y[tr]).predict(Fz[m])
        rows.append((itn, s, m.sum(), bas(y[m], pw), bas(y[m], pl)))
        print(f"{itn:9s} {s} {m.sum():4d}  {rows[-1][3]:.3f}  {rows[-1][4]:.3f}")
for itn in ("Graz", "BrainHero"):
    r = [x for x in rows if x[0] == itn]
    print(itn, "mean within %.3f  loso %.3f" % (np.mean([x[3] for x in r]), np.mean([x[4] for x in r])))
