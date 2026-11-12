"""OCU step 1 (ledger s58): how much class information do explicit ocular channels carry across people, with simple
models? Leave one person out; per-session centring (label-free). Usage: python studies/12nov/ocu_probe.py"""
import os, sys, warnings
import numpy as np
from scipy.signal import butter, sosfiltfilt, savgol_filter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
REPO = os.path.abspath(__file__ + "/../../..")
z = np.load(REPO + "/data/proteus_windows.npz", allow_pickle=True)
X, y, itf = z["X"].astype(np.float64), z["y"], z["interface"]; ch = [str(c) for c in z["ch_names"]]; ix = {c: i for i, c in enumerate(ch)}
per = np.array([s[-3:] for s in z["subject"]]); sid = np.char.add(per, z["session"].astype(str)); PEOPLE = sorted(set(per))
sel = lambda cs: X[:, [ix[c] for c in cs]].mean(1)  # noqa: E731
H = sel(("F5", "AF3", "F3")) - sel(("F6", "AF4", "F4")); V = sel(("AF3", "AF4")) - sel(("P1", "P2", "Pz")); H2 = sel(("FC5", "C5")) - sel(("FC6", "C6"))
Hs, Vs = savgol_filter(H, 25, 3, axis=-1), savgol_filter(V, 25, 3, axis=-1); MAG = np.sqrt(Hs ** 2 + Vs ** 2)
bins = lambda a, n: a[:, :a.shape[1] // n * n].reshape(len(a), n, -1).mean(-1)  # noqa: E731
dH, dV = np.diff(Hs, axis=-1), np.diff(Vs, axis=-1)
REPS = {"A: H, V in 16 bins (the shipped gaze view)": np.c_[bins(H, 16), bins(V, 16)],
        "B: A + magnitude": np.c_[bins(Hs, 16), bins(Vs, 16), bins(MAG, 16)],
        "C: B + second horizontal pair + velocity statistics": np.c_[bins(Hs, 16), bins(Vs, 16), bins(MAG, 16), bins(H2, 16), bins(np.abs(dH), 8), bins(np.abs(dV), 8),
                                                                    dH.std(1)[:, None], dV.std(1)[:, None], np.abs(dH).max(1)[:, None], np.abs(dV).max(1)[:, None]],
        "D: all 41 channels, 1 Hz waveform in 8 bins (core1's slow block)": bins(sosfiltfilt(butter(4, 1.0, "low", fs=120, output="sos"), X, axis=-1).reshape(len(X), -1), 1).reshape(len(X), 0) if False else
        sosfiltfilt(butter(4, 1.0, "low", fs=120, output="sos"), X, axis=-1)[:, :, 30::60].reshape(len(X), -1)}


def ba(r, pred):
    return 100 * np.mean([np.mean([np.mean(pred[(itf[r] == i) & (y[r] == c)] == c) for c in range(3)]) for i in np.unique(itf[r])])


print("class means of the session-centred, session-scaled horizontal and vertical traces (whole window), per person:")
hm, vm = H.mean(1), V.mean(1)
for s in np.unique(sid):
    m = sid == s; hm[m] = (hm[m] - hm[m].mean()) / hm[m].std(); vm[m] = (vm[m] - vm[m].mean()) / vm[m].std()
for p in PEOPLE:
    m = per == p; print(f"  {p}: H (left minus right) MI {hm[m & (y == 0)].mean():+.2f} SUB {hm[m & (y == 1)].mean():+.2f} WORD {hm[m & (y == 2)].mean():+.2f} | V (front minus back) MI {vm[m & (y == 0)].mean():+.2f} "
                        f"SUB {vm[m & (y == 1)].mean():+.2f} WORD {vm[m & (y == 2)].mean():+.2f}")
for name, F in REPS.items():
    F = F.copy()
    for s in np.unique(sid):
        m = sid == s; F[m] = (F[m] - F[m].mean(0)) / (F[m].std(0) + 1e-9)
    out = {}
    for tag, mk in (("LR", lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))), ("LDA", lambda: LDA(solver="lsqr", shrinkage="auto"))):
        v = []
        for p in PEOPLE:
            clf = mk().fit(F[per != p], y[per != p]); v.append(np.mean([ba(np.flatnonzero(sid == s), clf.predict(F[sid == s])) for s in np.unique(sid[per == p])]))
        out[tag] = v
    best = max(out, key=lambda k: np.mean(out[k]))
    print(f"RESULT {name}: LR {np.mean(out['LR']):.2f} LDA {np.mean(out['LDA']):.2f} | per person ({best}) " + " ".join(f"{p[1:]}:{x:.0f}" for p, x in zip(PEOPLE, out[best])), flush=True)
