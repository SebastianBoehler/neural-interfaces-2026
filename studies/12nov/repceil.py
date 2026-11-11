"""REPCEIL (ledger s36): within-session supervised ceilings in representations outside core1's PC space.
Leave-one-run-out per session, own labels only. Usage: python studies/12nov/repceil.py"""
import os, sys, warnings
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.linalg import eigh
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
REPO = os.path.abspath(__file__ + "/../../..")
z = np.load(REPO + "/data/proteus_windows.npz", allow_pickle=True)
X, y, itf, run, task = z["X"].astype(np.float64), z["y"], z["interface"], z["run"], z["task"]
sid = np.array([f"{s[-3:]}s{q}" for s, q in zip(z["subject"], z["session"])])
Xc = X - X.mean(1, keepdims=True)  # common-average reference
BANDS = [(1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 40), (40, 58), (4, 40), (8, 30)]
IU = np.triu_indices(41); WT = np.where(IU[0] == IU[1], 1.0, np.sqrt(2.0))
inp = dict(np.load(REPO + "/runs/replay_data/inputs.npz", allow_pickle=True))
assert (inp["idx|all_loso"] == np.arange(len(y))).all(); PC = inp["pc128|all_loso"]


def bp(lo, hi):
    return sosfiltfilt(butter(4, [lo, hi], btype="band", fs=120, output="sos"), Xc, axis=-1)


def covs(B):
    C = np.einsum("nct,ndt->ncd", B, B) / B.shape[-1]
    return C + 1e-3 * np.trace(C, axis1=1, axis2=2)[:, None, None] / 41 * np.eye(41)


def logm(C):
    w, V = np.linalg.eigh(C); return np.einsum("nij,nj,nkj->nik", V, np.log(w), V)


FB = [bp(*b) for b in BANDS]; CV = [covs(B) for B in FB]; LG = [logm(C) for C in CV]
SL = sosfiltfilt(butter(2, 4, btype="low", fs=120, output="sos"), X, axis=-1)[:, :, 7::15]  # 0.1-4 Hz, 8 per second


def tangent(m):
    out = []
    for L in LG:  # session-own log-Euclidean reference (label-free)
        out.append(((L[m] - L[m].mean(0))[:, IU[0], IU[1]]) * WT)
    return np.concatenate(out, 1)


def csp_feats(tr, te, ytr):
    ftr, fte = [], []
    for C in CV:
        Ctr, Cte = C[tr], C[te]; W = []
        for a, b in ((0, 1), (0, 2), (1, 2)):
            Ca, Cb = Ctr[ytr == a].mean(0), Ctr[ytr == b].mean(0); w, V = eigh(Ca, Ca + Cb); W.append(V[:, [0, 1, 2, -3, -2, -1]])
        W = np.concatenate(W, 1); f = lambda Q: np.log(np.einsum("cf,ncd,df->nf", W, Q, W))  # noqa: E731
        ftr.append(f(Ctr)); fte.append(f(Cte))
    return np.concatenate(ftr, 1), np.concatenate(fte, 1)


def lr(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=500))


REG = {"R0 core1 PCs": [0.01, 0.1, 1.0], "R1 own tangent": [1e-4, 1e-3, 1e-2], "R2 FBCSP": [0.1, 0.5, 0.9], "R3 slow": [0.3, 0.7, 0.95]}


def session(s):
    m = np.flatnonzero(sid == s); ys = y[m]; rk = np.char.add(task[m].astype(str), run[m].astype(str)); T = tangent(m)
    S = SL[m].reshape(len(m), -1); S = (S - S.mean(0)) / (S.std(0) + 1e-9); P0 = (PC[m] - PC[m].mean(0)) / (PC[m].std(0) + 1e-9)
    out = {(k, r): np.zeros((len(m), 3)) for k in REG for r in REG[k]}
    for u in np.unique(rk):
        te, tr = rk == u, rk != u
        for r in REG["R0 core1 PCs"]:
            out[("R0 core1 PCs", r)][te] = lr(r).fit(P0[tr], ys[tr]).predict_proba(P0[te])
        for r in REG["R1 own tangent"]:
            out[("R1 own tangent", r)][te] = lr(r).fit(T[tr], ys[tr]).predict_proba(T[te])
        a, b = csp_feats(m[tr], m[te], ys[tr])
        for r in REG["R2 FBCSP"]:
            out[("R2 FBCSP", r)][te] = LDA(solver="lsqr", shrinkage=r).fit(a, ys[tr]).predict_proba(b)
        for r in REG["R3 slow"]:
            out[("R3 slow", r)][te] = LDA(solver="lsqr", shrinkage=r).fit(S[tr], ys[tr]).predict_proba(S[te])
    return s, out


def ba(s, P):
    m = sid == s; p = P.argmax(1)
    return 100 * np.mean([np.mean([np.mean(p[(itf[m] == i) & (y[m] == k)] == k) for k in range(3)]) for i in np.unique(itf[m])])


res = dict(Parallel(n_jobs=6)(delayed(session)(s) for s in np.unique(sid))); SS = sorted(res)
best = {}
for k in REG:
    sc = {r: np.mean([ba(s, res[s][(k, r)]) for s in SS]) for r in REG[k]}; best[k] = max(sc, key=sc.get)
    print(f"{k:16s} " + " ".join(f"reg {r}: {v:.2f}" for r, v in sc.items()), flush=True)
fus = {"R4 = R1 x R3": ["R1 own tangent", "R3 slow"], "R5 = R0 x R1 x R3": ["R0 core1 PCs", "R1 own tangent", "R3 slow"],
       "R6 = R0 x R1 x R2 x R3": list(REG)}
tab = {k: [ba(s, res[s][(k, best[k])]) for s in SS] for k in REG}
for k, parts in fus.items():
    tab[k] = [ba(s, np.exp(sum(np.log(np.clip(res[s][(q, best[q])], 1e-4, 1)) for q in parts) / len(parts))) for s in SS]
print("\nsession  " + " ".join(f"{k[:2]:>6s}" for k in tab))
for i, s in enumerate(SS):
    print(f"{s}   " + " ".join(f"{tab[k][i]:6.1f}" for k in tab))
print("mean     " + " ".join(f"{np.mean(tab[k]):6.2f}" for k in tab))
bf = max(fus, key=lambda k: np.mean(tab[k])); d = np.array(tab[bf]) - np.array(tab["R0 core1 PCs"])
print(f"best fusion {bf}: {np.mean(tab[bf]):.2f}; minus R0 {d.mean():+.2f}; sessions with >= +5: {(d >= 5).sum()}/14")
