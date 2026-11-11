"""Identity-gate stress test with the PACKAGE gallery (all 14 public sessions): which inputs would be routed?"""
import glob, sys
import numpy as np
REPO = __import__("os").path.abspath(__file__ + "/../../../..")
sys.path.insert(0, REPO + "/experiments")
from bio_gate import Bio, fold_params
z = np.load(REPO + "/data/proteus_windows.npz", allow_pickle=True); X = z["X"]
PER = np.array([s[-3:] for s in z["subject"]]); STR = np.char.add(np.char.add(PER, "|"), z["session"].astype(str))
B0 = Bio.__new__(Bio); B0.win = np.hanning(120); FB = np.stack([B0.feat(x) for x in X])
P = fold_params(dict(np.load(sys.argv[1])), FB, STR, sorted(set(PER)), t0=10)


def run(W, n=200):
    b = Bio(P); best, who = -9.0, -1
    for x in W[:n]:
        b.step(x)
        if b.last and b.last[1] > best:
            best, who = b.last[1], b.last[0]
    return best, who


rng = np.random.default_rng(0); T = 150
real = X[STR == "P07|1"][:T]
syn = {"zeros": np.zeros((T, 41, 480), np.float32), "white noise 15 uV": rng.normal(0, 15, (T, 41, 480)).astype(np.float32),
       "common 50 Hz + noise": (30 * np.sin(2 * np.pi * 50 * np.arange(480) / 120)[None, None] + rng.normal(0, 5, (T, 41, 480))).astype(np.float32),
       "pink-ish noise": np.cumsum(rng.normal(0, 2, (T, 41, 480)), -1).astype(np.float32),
       "constant offset": np.full((T, 41, 480), 100.0, np.float32), "NaN-cleaned (zeros) half": np.concatenate([real[:75], np.zeros((75, 41, 480), np.float32)]),
       "real P07 x 10 amplitude": real * 10, "real P07 x 0.1": real * 0.1, "real P07 channels permuted": real[:, rng.permutation(41)],
       "real P07 time-reversed": real[:, :, ::-1].copy(), "real P07 one flat channel": np.where(np.arange(41)[None, :, None] == 5, 0, real).astype(np.float32),
       "real P07 + 20 uV white noise": real + rng.normal(0, 20, real.shape).astype(np.float32)}
print("synthetic / perturbed sessions (max score over 150 windows, matched person):")
for k, W in syn.items():
    s, j = run(W); print(f"  {k:32s} {s:+.3f} -> {sorted(set(PER))[j] if j >= 0 else None}")
print("public sessions in-sample (own sessions are in the gallery):")
for u in dict.fromkeys(STR):
    s, j = run(X[STR == u]); print(f"  {u} {s:+.3f} -> {sorted(set(PER))[j]}", end=";")
print()
for name, pat in (("dreyer2023", "/data/dreyer2023/*.npz"), ("scherer2015", "/data/scherer2015/*.npz")):
    out = []
    for fn in sorted(glob.glob(REPO + pat)):
        d = np.load(fn, allow_pickle=True)
        if "X" not in d.files:
            continue
        sub = d["subject"].astype(str); ses = d["session"].astype(str) if "session" in d.files else np.full(len(sub), "1")
        for key in dict.fromkeys(zip(sub, ses)):
            r = np.flatnonzero((sub == key[0]) & (ses == key[1]))
            if len(r) >= 30:
                out.append(run(d["X"][r])[0])
    out = np.sort(out)
    if len(out):
        print(f"{name}: {len(out)} sessions of other people; max score quantiles 50/90/95/99/max: {np.round(np.quantile(out, [.5, .9, .95, .99, 1]), 3).tolist()}; above 0.23: {int((out >= 0.23).sum())}, above 0.27: {int((out >= 0.27).sum())}, above 0.30: {int((out >= 0.30).sum())}")
