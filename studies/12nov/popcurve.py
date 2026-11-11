"""POPCURVE (ledger s39): core1 refitted on k other people; new-person BA against k. Usage: python studies/12nov/popcurve.py"""
import os, sys, time, warnings
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
REPO = os.path.abspath(__file__ + "/../../..")
sys.path[:0] = [REPO + "/experiments"]
import protocol  # noqa: E402
import ts_core2 as T  # noqa: E402

d = protocol.load(); t0 = time.time(); fe = T.Feats(d, T.parse("car_ov17_fs2_t0.1_le_k2")); y = d["y"]
print(f"features {fe.F.shape} in {time.time() - t0:.0f}s", flush=True)
per = np.array([s[-3:] for s in d["subject"]]); ses = d["session"].astype(str); itf = d["interface"]
PEOPLE = sorted(set(per)); KS, DRAWS = (2, 3, 5, 7), 4


def rebal(X, alpha=1.15, k=5.0, W=30):
    cs = np.vstack([np.zeros((1, 3)), np.cumsum(X, 0)]); n = np.arange(len(X)); lo = np.maximum(0, n - W)
    q = X / ((k / 3 + cs[n] - cs[lo]) / (k + (n - lo))[:, None]) ** alpha; return q / q.sum(1, keepdims=True)


def ba(r, pred):
    return 100 * np.mean([np.mean([np.mean(pred[(itf[r] == i) & (y[r] == c)] == c) for c in range(3)]) for i in np.unique(itf[r])])


def trial(p, train_people):
    idx = np.flatnonzero(np.isin(per, train_people)); pri = [fe.prior(b, idx) for b in range(len(fe.C))]; d2p = [q[idx].mean() for q in fe.d2]
    core = make_pipeline(StandardScaler(), LogisticRegression(C=10.0, max_iter=2000)).fit(fe.F[idx], y[idx]); a, b = [], []
    for s in sorted(set(ses[per == p])):
        r = np.flatnonzero((per == p) & (ses == s)); P = core.predict_proba(fe.causal(r, pri, d2p, None))
        a.append(ba(r, P.argmax(1))); b.append(ba(r, rebal(P).argmax(1)))
    return p, len(train_people), np.mean(a), np.mean(b)


rng = np.random.default_rng(0); jobs = []
for p in PEOPLE:
    oth = [q for q in PEOPLE if q != p]; jobs.append((p, oth))
    for k in KS:
        jobs += [(p, list(rng.choice(oth, k, replace=False))) for _ in range(DRAWS)]
t0 = time.time(); res = Parallel(n_jobs=5)(delayed(trial)(*j) for j in jobs); print(f"{len(jobs)} fits in {time.time() - t0:.0f}s", flush=True)
ks = sorted(set(r[1] for r in res)); tab = {(p, k): np.mean([[r[2], r[3]] for r in res if r[0] == p and r[1] == k], 0) for p in PEOPLE for k in ks}
print("people   " + " ".join(f"{p:>6s}" for p in PEOPLE) + "    mean")
for j, name in enumerate(("core", "rebal")):
    for k in ks:
        v = [tab[(p, k)][j] for p in PEOPLE]; print(f"{name:5s} k={k} " + " ".join(f"{x:6.1f}" for x in v) + f"  {np.mean(v):6.2f}")
for j, name in enumerate(("core", "rebal")):
    m = np.array([np.mean([tab[(p, k)][j] for p in PEOPLE]) for k in ks]); kk = np.array(ks, float); f = kk >= 3
    sl, ic = np.polyfit(np.log2(kk[f]), m[f], 1)
    print(f"{name}: per doubling {sl:+.2f}; 7 -> 9 {m[-1] - m[-2]:+.2f}; log-fit extrapolation k=10 {ic + sl * np.log2(10):.1f}, k=20 {ic + sl * np.log2(20):.1f}, k=40 {ic + sl * np.log2(40):.1f}")
