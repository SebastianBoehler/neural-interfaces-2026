"""s248 M3(ii) + M4 on the model-facing 120 Hz windows (data/proteus_windows.npz, 2,655 public windows).
M3(ii): max |Pearson r| between flattened windows of DIFFERENT runs (duplicate / overlapping trials), and between
        consecutive windows of the same run (time overlap; the cue gap is about 12 s, so none expected).
M4: oracle separability of runs inside a session: leave-one-out k-NN (k 1 and 5) on the true run label, features
    standardised within the session; (a) log-variance per channel in 6 bands (the s217 nuisance bands, unprojected);
    (b) hardware view: log power 45-58 Hz and 48-52 Hz per channel, channel correlations at 30-58 Hz;
    (c) a + b. Chance = the largest run's share of the session.
Usage: python studies/14nov/levers/run_fingerprint.py"""
import os
import numpy as np
from scipy.signal import butter, sosfiltfilt
REPO = os.path.abspath(__file__ + "/../../../..")
with np.load(REPO + "/data/proteus_windows.npz") as w:
    X, sub, ses, task, run = w["X"].astype(np.float32), w["subject"], w["session"], w["task"], w["run"]
SID = np.char.add(sub.astype(str), np.char.add("|", ses.astype(str))); RID = np.char.add(SID, np.char.add("|", run.astype(str)))
FS = 120.0
print(f"s248 run fingerprint: {len(X)} windows, {len(np.unique(SID))} sessions, {len(np.unique(RID))} runs")

# M3(ii) duplicates / overlap
F = X.reshape(len(X), -1).astype(np.float64); F -= F.mean(1, keepdims=True); F /= np.linalg.norm(F, axis=1, keepdims=True)
C = np.abs(F @ F.T); np.fill_diagonal(C, 0); same = RID[:, None] == RID[None, :]
print(f"M3 max |r| across different runs {C[~same].max():.3f} (pairs > 0.9: {(C[~same] > 0.9).sum() // 2}, > 0.5: {(C[~same] > 0.5).sum() // 2});"
      f" within a run {C[same].max():.3f}; 99.9th pct across runs {np.percentile(C[~same], 99.9):.3f}")
del F, C


def bandpow(lo, hi):
    sos = butter(4, [lo / (FS / 2), min(hi / (FS / 2), 0.99)], btype="band", output="sos")
    return sosfiltfilt(sos, X, axis=-1)


bands = [(4, 8), (8, 13), (13, 20), (20, 30), (30, 45), (45, 58)]
fa = np.concatenate([np.log(bandpow(lo, hi).var(-1) + 1e-6) for lo, hi in bands], 1)
hb = bandpow(30, 58); iu = np.triu_indices(41, 1)
cor = np.stack([np.corrcoef(h)[iu] for h in hb])
fb = np.concatenate([np.log(bandpow(45, 58).var(-1) + 1e-6), np.log(bandpow(48, 52).var(-1) + 1e-6), cor], 1)
feats = {"a log-var 6 bands": fa, "b hardware": fb, "c a+b": np.concatenate([fa, fb], 1)}


def knn_acc(Z, lab, k):
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9); D = ((Z[:, None] - Z[None]) ** 2).sum(-1); np.fill_diagonal(D, np.inf)
    nn = np.argsort(D, 1)[:, :k]; votes = lab[nn]
    pred = np.array([max(set(v), key=list(v).count) for v in votes]); return (pred == lab).mean()


for name, Fe in feats.items():
    a1, a5, ch = [], [], []
    for s in np.unique(SID):
        m = SID == s; lab = RID[m]; _, cnt = np.unique(lab, return_counts=True)
        a1.append(knn_acc(Fe[m], lab, 1)); a5.append(knn_acc(Fe[m], lab, 5)); ch.append(cnt.max() / cnt.sum())
    print(f"M4 {name:18s}: 1-NN {np.mean(a1):.3f} (range {min(a1):.2f}-{max(a1):.2f}) | 5-NN {np.mean(a5):.3f} | chance {np.mean(ch):.3f}")
