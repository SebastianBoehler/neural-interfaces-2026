"""Per-band template views (ledger s222, step 1b): the 41 x 41 tangent of each band at the causal session reference (test
sessions) as a TSEM view, so that tview.py measures per-band WITHIN-session template separability (the quantity the
template stage uses) next to bands.py's cross-person member accuracy. The member posteriors are copied from
runs/views/bands.npz so tview.py can compute the eye-net phi.
Usage: python studies/14nov/views/bandviews.py -> runs/views/bandviews.npz (<band>|view|<proto>, <band>|<proto>)"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib as V  # noqa: E402

BANDS = [(1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 45)]
T0 = time.time(); B = np.load(V.REPO + "/runs/views/bands.npz"); res = {}
for lo, hi in BANDS:
    C = V.cov(V.band(V.XCAR, lo, hi)); Lg = V.logm(C); name = f"b{lo:02d}-{hi:02d}"
    for proto in ("all_loso", "seen_session"):
        out = np.full((V.N, 861), np.nan, np.float32)
        for _, tr, te in V.protocol.folds(V.D, proto):
            _, R = V.references(C, V.session_runs(te), Lg[np.flatnonzero(tr)].mean(0))
            r = np.flatnonzero(te); out[r] = V.tangent(C[r], R[r])
        res[f"{name}|view|{proto}"], res[f"{name}|{proto}"] = out, B[f"{lo}-{hi}|{proto}"]
    print(f"{name} ({time.time() - T0:.0f}s)", flush=True)
np.savez_compressed(V.REPO + "/runs/views/bandviews.npz", **res)
print("saved")
