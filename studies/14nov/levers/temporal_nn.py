"""s248 M4d: does the hardware view (30-58 Hz channel correlations + high-band and line-band log power, standardised
within session) recover temporal neighbours in recording time? Per window, its nearest neighbour inside the session:
share in the same run, within +-1 / +-3 trials (recording order), chance levels from a random neighbour; Spearman of
view distance against time lag. Also the s217 log-variance view for reference.
Usage: python studies/14nov/levers/temporal_nn.py"""
import os
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import spearmanr
REPO = os.path.abspath(__file__ + "/../../../..")
with np.load(REPO + "/data/proteus_windows.npz") as w:
    X, sub, ses, run, onset = w["X"].astype(np.float32), w["subject"], w["session"], w["run"], w["onset"]
SID = np.char.add(sub.astype(str), np.char.add("|", ses.astype(str))); FS = 120.0


def bp(lo, hi):
    return sosfiltfilt(butter(4, [lo / (FS / 2), min(hi / (FS / 2), 0.99)], btype="band", output="sos"), X, axis=-1)


iu = np.triu_indices(41, 1); hb = bp(30, 58)
views = {"hardware": np.concatenate([np.log(bp(45, 58).var(-1) + 1e-6), np.log(bp(48, 52).var(-1) + 1e-6),
                                     np.stack([np.corrcoef(h)[iu] for h in hb])], 1),
         "log-var 6 bands": np.concatenate([np.log(bp(lo, hi).var(-1) + 1e-6) for lo, hi in
                                            [(4, 8), (8, 13), (13, 20), (20, 30), (30, 45), (45, 58)]], 1)}
rng = np.random.default_rng(0)
for name, Fe in views.items():
    sr, w1, w3, cr, c1, c3, rho = [], [], [], [], [], [], []
    for s in np.unique(SID):
        m = np.flatnonzero(SID == s); o = np.lexsort((onset[m], run[m])); m = m[o]  # recording order: run, then onset
        t = np.arange(len(m)); r = run[m]; Z = Fe[m]; Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
        D = ((Z[:, None] - Z[None]) ** 2).sum(-1); np.fill_diagonal(D, np.inf); nn = D.argmin(1)
        rnd = np.array([rng.choice(np.delete(t, i)) for i in t])
        sr.append((r[nn] == r).mean()); w1.append((np.abs(nn - t) <= 1).mean()); w3.append((np.abs(nn - t) <= 3).mean())
        cr.append((r[rnd] == r).mean()); c1.append((np.abs(rnd - t) <= 1).mean()); c3.append((np.abs(rnd - t) <= 3).mean())
        iu2 = np.triu_indices(len(m), 1); rho.append(spearmanr(D[iu2], np.abs(t[:, None] - t[None])[iu2])[0])
    print(f"{name:16s}: 1-NN same run {np.mean(sr):.3f} (chance {np.mean(cr):.3f}) | within +-1 trial {np.mean(w1):.3f} "
          f"(chance {np.mean(c1):.3f}) | within +-3 {np.mean(w3):.3f} (chance {np.mean(c3):.3f}) | Spearman(distance, lag) {np.mean(rho):+.3f}")
