"""CHAINSLOPE (ledger s48): core1 on k training people, then a core-only chain (rebalancing -> SessionEM -> TSEM).
Usage: python studies/12nov/chainslope.py"""
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
import session_em as SE  # noqa: E402
import tsem as TS  # noqa: E402
DATA = next(p for p in (os.environ.get("NI26_REPLAY_DATA", ""), REPO + "/runs/replay_data", REPO + "/runs/diag/replay_data") if p and os.path.isdir(p))
Z = np.load(DATA + "/inputs.npz", allow_pickle=True); GE = Z["GE|all_loso"]; assert (Z["idx|all_loso"] == np.arange(len(GE))).all()
d = protocol.load(); fe = T.Feats(d, T.parse("car_ov17_fs2_t0.1_le_k2")); y = d["y"]
per = np.array([s[-3:] for s in d["subject"]]); ses = d["session"].astype(str); itf = d["interface"]
PEOPLE = sorted(set(per)); KS, DRAWS, T0 = (3, 5, 7), 2, time.time()


def rebal(X, alpha=1.15, k=5.0, W=30):
    cs = np.vstack([np.zeros((1, 3)), np.cumsum(X, 0)]); n = np.arange(len(X)); lo = np.maximum(0, n - W)
    q = X / ((k / 3 + cs[n] - cs[lo]) / (k + (n - lo))[:, None]) ** alpha; return q / q.sum(1, keepdims=True)


def tsem(P, X, lam=0.15, n0=60, rho=0.7, shrink=0.7, m0=20, nmin=15):  # replay/grid.tsem_lik + apply_lam, one session
    out = P.copy(); lab, conf = P.argmax(1), P.max(1)
    for t in range(nmin, len(P)):
        A_, f = X[:t], X[t]; m, sd = A_.mean(0), np.maximum(A_.std(0), 1e-3); Zm, z = (A_ - m) / sd, (f - m) / sd
        q = max(2, int(rho * t / 3)); sel = [(ik := np.flatnonzero(lab[:t] == k))[np.argsort(-conf[ik])[:q]] for k in range(3)]
        if min(len(x) for x in sel) < 2:
            continue
        mu = np.stack([Zm[x].sum(0) / (len(x) + m0) for x in sel]); R = np.concatenate([Zm[x] - mu[k] for k, x in enumerate(sel)])
        Cv = np.cov(R.T); Cv = (1 - shrink) * Cv + shrink * np.eye(len(Cv)) * np.trace(Cv) / len(Cv)
        L = TS._lda(z[None], mu, np.linalg.inv(Cv))[0]; o = P[t] * L ** (lam * t / (t + n0)); out[t] = o / o.sum()
    return out


def ba(r, pred):
    return 100 * np.mean([np.mean([np.mean(pred[(itf[r] == i) & (y[r] == c)] == c) for c in range(3)]) for i in np.unique(itf[r])])


def trial(p, train_people):
    idx = np.flatnonzero(np.isin(per, train_people)); pri = [fe.prior(b, idx) for b in range(len(fe.C))]; d2p = [q[idx].mean() for q in fe.d2]
    core = make_pipeline(StandardScaler(), LogisticRegression(C=10.0, max_iter=2000)).fit(fe.F[idx], y[idx]); sc = core.steps[0][1]
    pca = PCA(128, random_state=0).fit(sc.transform(fe.F[idx])); a, b, c = [], [], []
    for s in sorted(set(ses[per == p])):
        r = np.flatnonzero((per == p) & (ses == s)); Fc = fe.causal(r, pri, d2p, None); P = core.predict_proba(Fc); Pr = rebal(P)
        Pe = SE.apply(GE[r], Pr, np.zeros(len(r), int), lam=0.5, n0=60, m0=20, shrink=0.7, nmin=10); Pt = tsem(Pe, pca.transform(sc.transform(Fc)))
        a.append(ba(r, P.argmax(1))); b.append(ba(r, Pe.argmax(1))); c.append(ba(r, Pt.argmax(1)))
    print(f"  fit {p} k={len(train_people)} core {np.mean(a):.1f} em {np.mean(b):.1f} chain {np.mean(c):.1f} ({time.time() - T0:.0f}s)", flush=True)
    return p, len(train_people), np.mean(a), np.mean(b), np.mean(c)


rng = np.random.default_rng(0); jobs = []
for p in PEOPLE:
    oth = [q for q in PEOPLE if q != p]; jobs.append((p, oth))
    for k in KS:
        jobs += [(p, list(rng.choice(oth, k, replace=False))) for _ in range(DRAWS)]
res = [trial(*j) for j in jobs]; ks = sorted(set(r[1] for r in res))
tab = {(p, k): np.mean([r[2:] for r in res if r[0] == p and r[1] == k], 0) for p in PEOPLE for k in ks}
for j, name in enumerate(("core", "em", "chain")):
    m = np.array([np.mean([tab[(p, k)][j] for p in PEOPLE]) for k in ks]); sl = np.polyfit(np.log2(np.array(ks, float)), m, 1)[0]
    print(f"RESULT {name:5s} " + " ".join(f"k={k}: {v:.2f}" for k, v in zip(ks, m)) + f" | per doubling {sl:+.2f}; 7 -> 9 {m[-1] - m[-2]:+.2f}", flush=True)
print("RESULT chain per person at k=9: " + " ".join(f"{p[1:]}:{tab[(p, 9)][2]:.1f}" for p in PEOPLE))
