"""Per-band member (ledger s222, step 1): CAR, band-pass, 41 x 41 trace-shrunk covariance, log-Euclidean tangent at the
session reference (training sessions whole-session, test sessions causal with the fold prior, k 2, recording order),
scaler + LR C 0.3; all_loso and seen_session. Also the full-41 sub-8 member (1-4 + 4-8 concatenated).
Usage: python studies/14nov/views/bands.py  -> runs/views/bands.npz (<band>|<proto> -> (N, 3) posteriors, NaN off test)"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib as V  # noqa: E402

BANDS = [(1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 45)]
T0 = time.time(); OUT = V.REPO + "/runs/views"; os.makedirs(OUT, exist_ok=True)
COV = {b: V.cov(V.band(V.XCAR, *b)) for b in BANDS}
print(f"covariances built ({time.time() - T0:.0f}s)", flush=True)


def feats(bands):
    def fn(tr, te):
        blocks = []
        for b in bands:
            C = COV[b]; Lg = V.logm(C)
            _, R = V.references(C, V.session_runs(te), Lg[tr].mean(0))
            blocks.append(V.tangent(C, R))
        return np.concatenate(blocks, 1)
    return fn


res = {}
for name, bands in [(f"{lo}-{hi}", [(lo, hi)]) for lo, hi in BANDS] + [("sub8full", BANDS[:2])]:
    P = V.lopo(feats(bands))
    res.update({f"{name}|{p}": v for p, v in P.items()})
    print(f"{name:9s} {V.summary(P)} ({time.time() - T0:.0f}s)", flush=True)
np.savez_compressed(OUT + "/bands.npz", **res)
print("saved", OUT + "/bands.npz")
