"""PERS (ledger s15): within-session proxy of the personal-head gain for all ten people, calibrated on P04 / P10.

Usage: python experiments/pers_proxy.py
"""
import os, sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT + "/experiments"]
import protocol  # noqa: E402
import ts_core2 as T  # noqa: E402

d = protocol.load(); fe = T.Feats(d, T.parse("car_ov17_fs2_t0.1_le_k2")); y = d["y"]
per = np.array([s[-3:] for s in d["subject"]]); ses = d["session"].astype(str); run = d["run"].astype(int)
ba = lambda yy, p: 100 * np.mean([np.mean(p[yy == k] == k) for k in range(3)])  # noqa: E731


def causal_std(Z):
    n = np.arange(1, len(Z) + 1)[:, None]; m = np.cumsum(Z, 0) / n; v = np.cumsum(Z ** 2, 0) / n - m ** 2
    return np.where(n >= 4, (Z - m) / np.sqrt(np.maximum(v, 1e-6)), Z - m)


def trial(others, pub_streams, test_streams):
    """others: training rows of other people; pub_streams / test_streams: lists of row arrays (each its own stream)."""
    idx = np.concatenate([others] + pub_streams); pri = [fe.prior(b, idx) for b in range(len(fe.C))]; d2p = [q[idx].mean() for q in fe.d2]
    cz = lambda r: fe.causal(r, pri, d2p, None)  # noqa: E731
    Fpub, Ftest = [cz(r) for r in pub_streams], [cz(r) for r in test_streams]
    core = make_pipeline(StandardScaler(), LogisticRegression(C=10.0, max_iter=2000)).fit(np.concatenate([fe.F[others]] + Fpub), y[idx])
    sc = core.steps[0][1]; pca = PCA(256, random_state=0).fit(sc.transform(fe.F[others]))
    view = lambda Fx: pca.transform(sc.transform(Fx))  # noqa: E731
    Zp = np.concatenate([(v - v.mean(0)) / np.maximum(v.std(0), 1e-6) for v in map(view, Fpub)])
    head = LogisticRegression(C=0.1, max_iter=3000).fit(Zp, np.concatenate([y[r] for r in pub_streams]))
    yt = np.concatenate([y[r] for r in test_streams]); Pc = np.concatenate([core.predict_proba(Fx) for Fx in Ftest])
    Ph = np.concatenate([head.predict_proba(causal_std(view(Fx))) for Fx in Ftest]); mix = (Pc + 3 * Ph) / 4
    return ba(yt, Pc.argmax(1)), ba(yt, Ph.argmax(1)), ba(yt, mix.argmax(1)), len(yt)


print("person  kind            core1  head   mix   gain  n_test", flush=True)
res = {}
for p in sorted(set(per)):
    others = np.flatnonzero(per != p); s1 = (per == p) & (ses == "1"); runs = sorted(set(run[s1]))
    A, B = np.flatnonzero(s1 & np.isin(run, runs[:3])), np.flatnonzero(s1 & np.isin(run, runs[3:]))
    c, h, m, n = trial(others, [A], [B]); res[p] = (c, m - c); print(f"{p}     within-session  {c:5.1f} {h:5.1f} {m:5.1f} {m - c:+6.1f}  {n}", flush=True)
    later = sorted(set(ses[per == p]) - {"1"})
    if later:
        c2, h2, m2, n2 = trial(others, [np.flatnonzero(s1)], [np.flatnonzero((per == p) & (ses == s)) for s in later])
        print(f"{p}     cross-session   {c2:5.1f} {h2:5.1f} {m2:5.1f} {m2 - c2:+6.1f}  {n2}   (real: session 1 -> sessions {','.join(later)})", flush=True)
        ch, hh, mh, nh = trial(others, [A], [np.flatnonzero((per == p) & (ses == s)) for s in later])
        print(f"{p}     cross, half pub {ch:5.1f} {hh:5.1f} {mh:5.1f} {mh - ch:+6.1f}  {nh}   (runs 1-3 of session 1 -> sessions {','.join(later)})", flush=True)
g = np.array([v[1] for v in res.values()]); w = np.array([100 - v[0] for v in res.values()])
from scipy.stats import spearmanr  # noqa: E402
print(f"mean proxy gain {g.mean():+.2f}; people with gain >= half of P04's: {int((g >= res['P04'][1] / 2).sum())}/10; spearman(gain, weakness) {spearmanr(g, w)[0]:+.2f}")
