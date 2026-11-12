"""ROTAUG (ledger s46b): core1 with rotated virtual copies of every training session. Usage: python studies/12nov/rotaug.py"""
import os, sys, time, warnings
import numpy as np
from scipy.linalg import expm
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
REPO = os.path.abspath(__file__ + "/../../..")
sys.path[:0] = [REPO + "/experiments"]
import protocol  # noqa: E402
import ts_core2 as T  # noqa: E402
d = protocol.load(); fe = T.Feats(d, T.parse("car_ov17_fs2_t0.1_le_k2")); y = d["y"]; F32 = fe.F.astype(np.float32)
per = np.array([s[-3:] for s in d["subject"]]); ses = d["session"].astype(str); itf = d["interface"]; sid = np.char.add(per, ses)
PEOPLE = sorted(set(per)); NB = len(fe.C); IU = np.triu_indices(41); WT = np.where(IU[0] == IU[1], 1.0, np.sqrt(2.0)).astype(np.float32)
COPIES = int(os.environ.get("COPIES", 2)); T0 = time.time()


def rotated(rows, R):
    out = F32[rows].copy(); R = R.astype(np.float32)
    for b in range(NB):
        sl = fe.slices[b]; L = np.zeros((len(rows), 41, 41), np.float32); V = out[:, sl] / WT
        L[:, IU[0], IU[1]] = V; L[:, IU[1], IU[0]] = V; L = R @ L @ R.T; out[:, sl] = L[:, IU[0], IU[1]] * WT
    return out


def ba(r, pred):
    return 100 * np.mean([np.mean([np.mean(pred[(itf[r] == i) & (y[r] == c)] == c) for c in range(3)]) for i in np.unique(itf[r])])


def fold(p, eps):
    idx = np.flatnonzero(per != p); Xtr, ytr = [F32[idx]], [y[idx]]; rng = np.random.default_rng(hash((p, eps)) % 2 ** 32)
    if eps > 0:
        for s in sorted(set(sid[idx])):
            r = np.flatnonzero(sid == s)
            for _ in range(COPIES):
                G = rng.standard_normal((41, 41)); Aa = (G - G.T) / np.sqrt(2 * 41); Xtr.append(rotated(r, expm(eps * Aa))); ytr.append(y[r])
    core = make_pipeline(StandardScaler(), LogisticRegression(C=10.0, max_iter=2000)).fit(np.concatenate(Xtr), np.concatenate(ytr))
    pri = [fe.prior(b, idx) for b in range(NB)]; d2p = [q[idx].mean() for q in fe.d2]; a = []
    for s in sorted(set(ses[per == p])):
        r = np.flatnonzero((per == p) & (ses == s)); a.append(ba(r, core.predict_proba(fe.causal(r, pri, d2p, None)).argmax(1)))
    return np.mean(a)


res = {}
for eps in [float(e) for e in os.environ.get("EPS", "0,0.1,0.2,0.4").split(",")]:
    res[eps] = []
    for p in PEOPLE:
        res[eps].append(fold(p, eps)); print(f"  eps {eps} {p} {res[eps][-1]:.1f} ({time.time() - T0:.0f}s)", flush=True)
    v = np.array(res[eps]); b = np.array(res[0.0]) if 0.0 in res else v
    print(f"eps {eps}: mean {v.mean():.2f} ({v.mean() - b.mean():+.2f} vs eps 0; {int((v > b + 1e-9).sum())} of 10 people up) | " + " ".join(f"{p[1:]}:{x:.1f}" for p, x in zip(PEOPLE, v)), flush=True)
