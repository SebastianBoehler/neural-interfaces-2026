"""PEFF (ledger s49): people-efficient estimators and noise for core1, leave-one-person-out. Usage: python studies/12nov/peff.py"""
import os, sys, time, warnings
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
REPO = os.path.abspath(__file__ + "/../../..")
sys.path[:0] = [REPO + "/experiments"]
import protocol  # noqa: E402
import ts_core2 as T  # noqa: E402
d = protocol.load(); fe = T.Feats(d, T.parse("car_ov17_fs2_t0.1_le_k2")); y = d["y"]
per = np.array([s[-3:] for s in d["subject"]]); ses = d["session"].astype(str); itf = d["interface"]
PEOPLE = sorted(set(per)); T0 = time.time(); K = 512


def ba(r, pred):
    return 100 * np.mean([np.mean([np.mean(pred[(itf[r] == i) & (y[r] == c)] == c) for c in range(3)]) for i in np.unique(itf[r])])


def lr(X, yy, C, w=None):
    return LogisticRegression(C=C, max_iter=1000).fit(X, yy, sample_weight=w)


def relda(X, yy, pp, tau, shrink=0.1):
    """Gaussian classes with person-equal means and covariance S_within + tau * S_between-person; returns (W, b)."""
    people = sorted(set(pp)); M = np.stack([[X[(pp == q) & (yy == k)].mean(0) for k in range(3)] for q in people]); mu = M.mean(0)
    R = np.concatenate([X[(pp == q) & (yy == k)] - M[i, k] for i, q in enumerate(people) for k in range(3)]); Sw = R.T @ R / len(R)
    Dv = (M - mu).reshape(-1, X.shape[1]); S = Sw + tau * (Dv.T @ Dv) / len(Dv); S = (1 - shrink) * S + shrink * np.eye(len(S)) * np.trace(S) / len(S)
    W = np.linalg.solve(S, mu.T).T; return W, -0.5 * np.einsum("kd,kd->k", W, mu)


def fold(p):
    tr = np.flatnonzero(per != p); pp, yt = per[tr], y[tr]; sc = StandardScaler().fit(fe.F[tr]); Zf = sc.transform(fe.F[tr])
    pca = PCA(K, random_state=0).fit(Zf); Z = pca.transform(Zf); sd = Z.std(0); Z = Z / sd
    pri = [fe.prior(b, tr) for b in range(len(fe.C))]; d2p = [q[tr].mean() for q in fe.d2]; rng = np.random.default_rng(PEOPLE.index(p)); out = {}
    tests = []
    for s in sorted(set(ses[per == p])):
        r = np.flatnonzero((per == p) & (ses == s)); Ff = sc.transform(fe.causal(r, pri, d2p, None)); tests.append((r, Ff, pca.transform(Ff) / sd))
    sc_ = lambda f, full=False: np.mean([ba(r, f(Ff if full else Zt)) for r, Ff, Zt in tests])  # noqa: E731
    for C in (0.01, 0.1, 1.0):
        m = lr(Z, yt, C); out[f"E1 lr C{C}"] = sc_(lambda A: m.predict(A))
    wb = np.array([1.0 / np.sum(pp == q) for q in pp]); wb *= len(pp) / wb.sum(); m = lr(Z, yt, 0.1, wb); out["E2 balanced C0.1"] = sc_(lambda A: m.predict(A))
    for tau in (0, 30, 100, 300):
        W, b = relda(Z, yt, pp, tau); out[f"E3 relda tau{tau}"] = sc_(lambda A: (A @ W.T + b).argmax(1))
    people = sorted(set(pp)); Mf = np.stack([[Zf[(pp == q) & (yt == k)].mean(0) for k in range(3)] for q in people])
    tt = np.max([np.abs((c := Mf[:, a] - Mf[:, b_]).mean(0)) / (c.std(0, ddof=1) / np.sqrt(len(people)) + 1e-9) for a, b_ in ((0, 1), (0, 2), (1, 2))], 0)
    for mfeat in (2000, 8000):
        sel = np.argsort(-tt)[:mfeat]; m = LogisticRegression(C=10.0, max_iter=2000).fit(Zf[:, sel], yt); out[f"E4 consistent m{mfeat}"] = sc_(lambda A: m.predict(A[:, sel]), True)
    bags = [lr(Z[np.isin(pp, q)], yt[np.isin(pp, q)], 0.1) for q in (rng.choice(people, 6, replace=False) for _ in range(12))]
    out["E5 people-bagging"] = sc_(lambda A: np.mean([m.predict_proba(A) for m in bags], 0).argmax(1))
    w = np.ones(len(pp))
    for _ in range(8):
        m = lr(Z, yt, 0.1, w); P = m.predict_proba(Z); loss = {q: -np.mean(np.log(P[pp == q, yt[pp == q]] + 1e-9)) for q in people}
        g = np.array([np.exp(loss[q]) for q in pp]); w = g * len(pp) / g.sum()
    out["E6 group DRO"] = sc_(lambda A: m.predict(A))
    for sg in (0.5, 1.0):
        Xa = np.concatenate([Z] + [Z + sg * rng.standard_normal(Z.shape) for _ in range(2)]); m = lr(Xa, np.tile(yt, 3), 0.1); out[f"E7 noise {sg}"] = sc_(lambda A: m.predict(A))
    Mz = np.stack([[Z[(pp == q) & (yt == k)].mean(0) for k in range(3)] for q in people]); sb = Mz.std(0, ddof=1)  # (3, K)
    for g_ in (1.0, 2.0):
        Xa = np.concatenate([Z] + [Z + g_ * sb[yt] * rng.standard_normal(Z.shape) for _ in range(2)])
        m = lr(Xa, np.tile(yt, 3), 0.1); out[f"E8 person-jitter {g_}"] = sc_(lambda A: m.predict(A))
    print(f"  {p} done ({time.time() - T0:.0f}s) " + " ".join(f"{v:.0f}" for v in out.values()), flush=True)
    return out


res = [fold(p) for p in PEOPLE]
for k in res[0]:
    v = np.array([r[k] for r in res]); print(f"RESULT {k:22s} mean {v.mean():.2f} | " + " ".join(f"{p[1:]}:{x:.1f}" for p, x in zip(PEOPLE, v)), flush=True)
