"""Lane R post-hoc oracle O3 (added after seeing O1/O2 far below stock on P04/P10; NOT in the registration):
person-only LR on 1-30 Hz time-domain windows (decimated to 30 Hz, per-channel z-score, whitened by the enrolment EA matrix), C fixed 0.01."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle import isqrt, ba, ROOT
SOS = butter(4, (1, 30), "bandpass", fs=120, output="sos")


def prep(X, W):
    F = sosfiltfilt(SOS, np.nan_to_num(np.asarray(X, np.float64)), axis=-1)[:, :, ::4]
    F = np.einsum("ij,njt->nit", W, F)
    return F.reshape(len(F), -1)


def run(subj, sess, y, X, people, tests):
    out = {}
    for p in people:
        re = np.flatnonzero((subj == p) & (sess == "1"))
        Fr = sosfiltfilt(SOS, np.nan_to_num(X[re].astype(np.float64)), axis=-1)
        C = np.einsum("nct,ndt->cd", Fr, Fr) / Fr.shape[0] / Fr.shape[-1]
        W = isqrt(C + 1e-3 * np.trace(C) / 41 * np.eye(41))
        Ftr = prep(X[re], W); mu, sd = Ftr.mean(0), np.maximum(Ftr.std(0), 1e-6)
        m = LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced").fit((Ftr - mu) / sd, y[re])
        for v in tests:
            rt = np.flatnonzero((subj == p) & (sess == v))
            if len(rt): out[f"{p}|{v}"] = ba(y[rt], m.predict((prep(X[rt], W) - mu) / sd)); print(p, v, round(out[f"{p}|{v}"], 1), flush=True)
    return out


z = np.load(ROOT / "data/proteus_windows.npz", allow_pickle=True)
res = {"proteus": run(z["subject"], z["session"].astype(str), z["y"], z["X"], ["Dreyer2026Proteus/P04", "Dreyer2026Proteus/P10"], ["2", "3"])}
s = np.load(ROOT / "data/scherer2015/windows.npz", allow_pickle=True)
res["scherer"] = run(s["subject"], s["session"].astype(str), s["y"], s["X"], sorted(set(s["subject"])), ["2"])
json.dump(res, open(sys.argv[1], "w"), indent=1)
