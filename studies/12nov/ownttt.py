"""OWNTTT (ledger s38): session-own members (FBCSP, own tangent space, slow) refitted inside the session on true or
chain labels, fused into the v3 chain output. Usage: python studies/12nov/ownttt.py"""
import os, sys, warnings
import numpy as np
from joblib import Parallel, delayed
REPO = os.path.abspath(__file__ + "/../../..")
sys.path[:0] = [REPO + "/replay", REPO + "/studies/12nov"]
import env  # noqa: E402
import ownspace as O  # noqa: E402
from scipy.linalg import eigh  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
warnings.filterwarnings("ignore")
A = env.A; SC = A.SC; Z = A.L.Z
T0, STEP, RHO = 30, 5, 0.7
PAIRS = ((0, 1), (0, 2), (1, 2))


def fit_block(g, t, lab, sel):
    """g: global rows of the session in order; fit on g[sel] (subset of the first t), predict g[t:t+STEP]."""
    past, blk = g[:t], g[t:t + STEP]; out = {}
    Cw_tr, Cw_te = [], []
    for C in O.CV:
        W = O._isqrt(C[past].mean(0)); Cw_tr.append(W @ C[g[sel]] @ W); Cw_te.append(W @ C[blk] @ W)
    yt = lab[sel]
    out["T"] = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=300)).fit(O.tang(Cw_tr), yt).predict_proba(O.tang(Cw_te))
    ftr, fte = [], []
    for a_, b_ in zip(Cw_tr, Cw_te):
        Wc = np.concatenate([eigh(a_[yt == i].mean(0), a_[yt == i].mean(0) + a_[yt == j].mean(0))[1][:, [0, 1, 2, -3, -2, -1]] for i, j in PAIRS], 1)
        ftr.append(np.log(np.einsum("cf,ncd,df->nf", Wc, a_, Wc))); fte.append(np.log(np.einsum("cf,ncd,df->nf", Wc, b_, Wc)))
    out["C"] = LDA(solver="lsqr", shrinkage=0.5).fit(np.concatenate(ftr, 1), yt).predict_proba(np.concatenate(fte, 1))
    m, sd = O.SL[past].mean(0), O.SL[past].std(0) + 1e-9
    out["S"] = LDA(solver="lsqr", shrinkage=0.7).fit((O.SL[g[sel]] - m) / sd, yt).predict_proba((O.SL[blk] - m) / sd)
    return out


def session(g, P, ytrue, mode):
    n = len(g); H = {k: np.full((n, 3), np.nan) for k in "TCS"}; pl_, cf = P.argmax(1), P.max(1)
    for t in range(T0, n, STEP):
        if mode == "true":
            lab, sel = ytrue, np.arange(t)
        else:
            lab = pl_; q = max(4, int(RHO * t / 3)); sel = np.sort(np.concatenate([(ik := np.flatnonzero(pl_[:t] == k))[np.argsort(-cf[ik])[:q]] for k in range(3)]))
        if min(np.sum(lab[sel] == k) for k in range(3)) < 4:
            continue
        o = fit_block(g, t, lab, sel)
        for k in "TCS":
            H[k][t:t + STEP] = o[k]
    return H


def fuse(P, H, tt, lam, keys):
    out = P.copy(); m = ~np.isnan(H[keys[0]][:, 0]); e = lam * tt[m] / (tt[m] + 60)
    L = np.log(np.clip(P[m], 1e-9, 1)) + e[:, None] * sum(np.log(np.clip(H[k][m], 1e-3, 1)) for k in keys) / len(keys)
    q = np.exp(L - L.max(1, keepdims=True)); out[m] = q / q.sum(1, keepdims=True); return out


for bn in os.environ.get("BASES", "B2_pool,B2_s345").split(","):
    base = A.system(bn, A.FAMS)["out"]; print(SC.absline(bn + " chain", base), flush=True)
    for mode in ("true", "chain"):
        HH, TT = {}, {}
        for p in A.PROTOS:
            ss, idx, y = Z[f"sess|{p}"], Z[f"idx|{p}"], Z[f"y|{p}"]; U = np.unique(ss)
            res = Parallel(n_jobs=6)(delayed(session)(idx[ss == s], base[p][ss == s], y[ss == s], mode) for s in U)
            HH[p] = {k: np.full((len(y), 3), np.nan) for k in "TCS"}; TT[p] = np.zeros(len(y))
            for s, h in zip(U, res):
                r = np.flatnonzero(ss == s); TT[p][r] = np.arange(len(r))
                for k in "TCS":
                    HH[p][k][r] = h[k]
            ok = ~np.isnan(HH[p]["T"][:, 0])
            print(f"   {mode} labels, {p}: members alone, window accuracy " + " ".join(f"{k} {100 * np.mean(HH[p][k][ok].argmax(1) == y[ok]):.1f}" for k in "TCS")
                  + f" | T+C+S {100 * np.mean(sum(np.log(np.clip(HH[p][k][ok], 1e-3, 1)) for k in 'TCS').argmax(1) == y[ok]):.1f} | chain {100 * np.mean(base[p][ok].argmax(1) == y[ok]):.1f}", flush=True)
        for keys in ("TC", "TCS"):
            for lam in (0.25, 0.5, 1.0):
                arm = {p: fuse(base[p], HH[p], TT[p], lam, keys) for p in A.PROTOS}
                print("   " + SC.line(f"{mode} labels, {keys}, lam {lam}", SC.compare(arm, base)), flush=True)
