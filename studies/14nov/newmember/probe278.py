"""Ledger s278 step 2: Shin 2018 WG vs baseline, within-corpus LOSO probe. Member as s264 (core feature set: 17 CAR bands, shrunk log-Euclidean covariance,
slow block; every person's recording centred at its full mean (primary; all one session, WG and baseline mixed) or by the causal running mean (secondary, K0 5);
standardised, LR C 0.05), computed on the 16 RECORDED channels only (the 25 interpolated ones are dropped). BA per person; 26 people.
Usage: python probe278.py  -> evidence/newmember_15nov/s278_probe.txt"""
import os, sys, numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "3")
REPO = os.path.abspath(__file__ + "/../../../.."); sys.path.insert(0, REPO + "/studies/14nov/extharness"); sys.path.insert(0, REPO + "/experiments")
import corpus as CO  # noqa: E402
from harness import K0  # noqa: E402
from features import slow_waveform  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402

z = np.load(REPO + "/data/shin2018/windows.npz", allow_pickle=True); names = [str(c) for c in z["ch_names"]]; rec = [str(c) for c in z["recorded"]]
idx = [names.index(c) for c in rec]; X = z["X"][:, idx].astype(np.float32); y = z["y"].astype(int); p = z["subject"].astype(str); onset = z["onset"]
assert X.shape[1] == 16 and len(set(p)) == 26 and set(np.unique(y)) == {0, 1} and (np.bincount(y) == len(y) // 2).all() and len(set(z["session"])) == 1
assert all(np.all(np.diff(onset[p == q]) >= 0) for q in set(p)), "windows not in time order within person (causal centring needs it)"
V = CO.tangent(X, CO.OV17); Xc = X - X.mean(1, keepdims=True); V = np.concatenate([V, slow_waveform(Xc, cutoff=1.0)], 1).astype(np.float64)
print("features", V.shape, flush=True)


def center(causal):
    out = V.copy()
    for q in dict.fromkeys(p):
        r = np.flatnonzero(p == q); out[r] -= (np.cumsum(out[r], 0) / (K0 + 1 + np.arange(len(r)))[:, None]) if causal else out[r].mean(0)
    return out


Vc, Vk = center(False), center(True); res = []
for q in dict.fromkeys(p):
    te = p == q; tr = ~te; mu, sd = Vc[tr].mean(0), Vc[tr].std(0) + 1e-6
    clf = LogisticRegression(C=0.05, max_iter=500, tol=1e-4).fit((Vc[tr] - mu) / sd, y[tr])
    res.append((q, te.sum(), 100 * balanced_accuracy_score(y[te], clf.predict((Vc[te] - mu) / sd)), 100 * balanced_accuracy_score(y[te], clf.predict((Vk[te] - mu) / sd))))
    print(res[-1], flush=True)
a = np.array([r[2] for r in res]); b = np.array([r[3] for r in res]); rng = np.random.default_rng(0); bs = np.sort([rng.choice(a, len(a)).mean() for _ in range(10000)])
L = ["# s278 step 2: Shin2018 WG vs baseline, LOSO inside the corpus, 16 recorded channels, core member, LR C 0.05", "person windows BA_full_centring BA_causal_centring"] + [f"{q} {n} {x:.1f} {c:.1f}" for q, n, x, c in res]
L.append(f"MEAN BA {a.mean():.1f} 80% bootstrap interval over people [{bs[1000]:.1f}, {bs[9000]:.1f}] people 10-90% [{np.percentile(a, 10):.1f}, {np.percentile(a, 90):.1f}] min {a.min():.1f} max {a.max():.1f} | causal-centring mean {b.mean():.1f} | people >60: {(a > 60).sum()}/26")
open(REPO + "/evidence/newmember_15nov/s278_probe.txt", "w").write("\n".join(L) + "\n"); print(L[-1])
