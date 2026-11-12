"""OCU step 2b (ledger s58): eye-movement features fused into core1 at feature level (a weighted extra block).
Usage: python studies/12nov/corefuse.py"""
import os, sys, time, warnings
import numpy as np
from scipy.signal import savgol_filter
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
REPO = os.path.abspath(__file__ + "/../../..")
sys.path[:0] = [REPO + "/experiments"]
import protocol  # noqa: E402
import ts_core2 as T  # noqa: E402
d = protocol.load(); fe = T.Feats(d, T.parse("car_ov17_fs2_t0.1_le_k2")); y = d["y"]; X = d["X"].astype(np.float64); ix = {str(c): i for i, c in enumerate(d["ch_names"])}
per = np.array([s[-3:] for s in d["subject"]]); ses = d["session"].astype(str); itf = d["interface"]; sid = np.char.add(per, ses); PEOPLE = sorted(set(per)); T0 = time.time()
sel = lambda cs: X[:, [ix[c] for c in cs]].mean(1)  # noqa: E731
bins = lambda a, n: a[:, :a.shape[1] // n * n].reshape(len(a), n, -1).mean(-1)  # noqa: E731
H = savgol_filter(sel(("F5", "AF3", "F3")) - sel(("F6", "AF4", "F4")), 25, 3, axis=-1); V = savgol_filter(sel(("AF3", "AF4")) - sel(("P1", "P2", "Pz")), 25, 3, axis=-1)
H2 = sel(("FC5", "C5")) - sel(("FC6", "C6")); dH, dV = np.diff(H, axis=-1), np.diff(V, axis=-1); lg = lambda a: np.log(a + 1e-3)  # noqa: E731
OC = np.c_[bins(H, 16), bins(V, 16), bins(np.sqrt(H ** 2 + V ** 2), 16), bins(H2, 16), lg(bins(np.abs(dH), 8)), lg(bins(np.abs(dV), 8)), lg(dH.std(1))[:, None], lg(dV.std(1))[:, None],
           lg(np.abs(dH).max(1))[:, None], lg(np.abs(dV).max(1))[:, None]]
OCw = OC.copy()  # training form: whole-session z-score; test form: causal z-score
for s in np.unique(sid):
    m = sid == s; OCw[m] = (OC[m] - OC[m].mean(0)) / (OC[m].std(0) + 1e-9)


def causal_z(F):
    n = np.arange(1, len(F) + 1)[:, None]; m = np.cumsum(F, 0) / n; v = np.cumsum(F ** 2, 0) / n - m ** 2
    return np.where(n >= 4, (F - m) / np.sqrt(np.maximum(v, 1e-9)), 0.0)


def ba(r, pred):
    return 100 * np.mean([np.mean([np.mean(pred[(itf[r] == i) & (y[r] == c)] == c) for c in range(3)]) for i in np.unique(itf[r])])


GS = (0.0, 3.0, 10.0, 30.0); res = {g: [] for g in GS}
for p in PEOPLE:
    idx = np.flatnonzero(per != p); sc = StandardScaler().fit(fe.F[idx]); Ztr = sc.transform(fe.F[idx]); pri = [fe.prior(b, idx) for b in range(len(fe.C))]; d2p = [q[idx].mean() for q in fe.d2]
    tests = [(r, sc.transform(fe.causal(r, pri, d2p, None)), causal_z(OC[r])) for r in (np.flatnonzero((per == p) & (ses == s)) for s in sorted(set(ses[per == p])))]
    for g in GS:
        m = LogisticRegression(C=10.0, max_iter=2000).fit(np.c_[Ztr, g * OCw[idx]] if g else Ztr, y[idx])
        res[g].append(np.mean([ba(r, m.predict(np.c_[Zt, g * Oc] if g else Zt)) for r, Zt, Oc in tests]))
    print(f"  {p} " + " ".join(f"g{g:g}:{res[g][-1]:.1f}" for g in GS) + f" ({time.time() - T0:.0f}s)", flush=True)
for g in GS:
    v = np.array(res[g]); print(f"RESULT block weight {g:g}: mean {v.mean():.2f} ({v.mean() - np.mean(res[0.0]):+.2f}; {int((v > np.array(res[0.0]) + 1e-9).sum())} of 10 up) | " + " ".join(f"{q[1:]}:{x:.1f}" for q, x in zip(PEOPLE, v)), flush=True)
