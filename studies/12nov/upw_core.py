"""UPW (ledger s50), part 1: core1 refitted on the seen folds with the returning person's own rows up-weighted.
Writes runs/upw_core.npz: P|<w> (n_windows, 3), NaN outside the seen test sessions. Usage: python studies/12nov/upw_core.py"""
import os, sys, time, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
REPO = os.path.abspath(__file__ + "/../../..")
sys.path[:0] = [REPO + "/experiments"]
import protocol  # noqa: E402
import ts_core2 as T  # noqa: E402
d = protocol.load(); fe = T.Feats(d, T.parse("car_ov17_fs2_t0.1_le_k2")); y = d["y"]
per = np.array([s[-3:] for s in d["subject"]]); ses = d["session"].astype(str); T0 = time.time()
out = {}
for w in (1, 3, 5, 10, 20):
    P = np.full((len(y), 3), np.nan)
    for p in ("P04", "P10"):
        idx = np.flatnonzero((per != p) | (ses == "1")); sw = np.where(per[idx] == p, float(w), 1.0)
        pri = [fe.prior(b, idx) for b in range(len(fe.C))]; d2p = [q[idx].mean() for q in fe.d2]
        core = make_pipeline(StandardScaler(), LogisticRegression(C=10.0, max_iter=2000)).fit(fe.F[idx], y[idx], logisticregression__sample_weight=sw)
        for s in sorted(set(ses[per == p]) - {"1"}):
            r = np.flatnonzero((per == p) & (ses == s)); P[r] = core.predict_proba(fe.causal(r, pri, d2p, None))
        print(f"w {w} {p}: accuracy on later sessions {100 * np.mean(P[(per == p) & (ses != '1')].argmax(1) == y[(per == p) & (ses != '1')]):.1f} ({time.time() - T0:.0f}s)", flush=True)
    out[f"P|{w}"] = P
os.makedirs(REPO + "/runs", exist_ok=True); np.savez_compressed(REPO + "/runs/upw_core.npz", **out); print("RESULT saved runs/upw_core.npz", flush=True)
