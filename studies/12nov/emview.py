"""EMVIEW (ledger s58, step 2): SessionEM's ocular view extended by a second horizontal pair and velocity statistics.
Usage: python studies/12nov/emview.py"""
import os, sys
import numpy as np
from scipy.signal import savgol_filter
REPO = os.path.abspath(__file__ + "/../../..")
sys.path[:0] = [REPO + "/replay"]
import env  # noqa: E402
A = env.A; SC = A.SC; Z = A.L.Z
z = np.load(REPO + "/data/proteus_windows.npz", allow_pickle=True); X = z["X"].astype(np.float64); ix = {str(c): i for i, c in enumerate(z["ch_names"])}
sel = lambda cs: X[:, [ix[c] for c in cs]].mean(1)  # noqa: E731
bins = lambda a, n: a[:, :a.shape[1] // n * n].reshape(len(a), n, -1).mean(-1)  # noqa: E731
H = savgol_filter(sel(("F5", "AF3", "F3")) - sel(("F6", "AF4", "F4")), 25, 3, axis=-1); V = savgol_filter(sel(("AF3", "AF4")) - sel(("P1", "P2", "Pz")), 25, 3, axis=-1)
H2 = sel(("FC5", "C5")) - sel(("FC6", "C6")); dH, dV = np.diff(H, axis=-1), np.diff(V, axis=-1); lg = lambda a: np.log(a + 1e-3)  # noqa: E731
PAIR = bins(H2, 16); VEL = np.c_[lg(bins(np.abs(dH), 8)), lg(bins(np.abs(dV), 8)), lg(dH.std(1))[:, None], lg(dV.std(1))[:, None], lg(np.abs(dH).max(1))[:, None], lg(np.abs(dV).max(1))[:, None]]
REF = {p: Z[f"GE|{p}"].copy() for p in A.PROTOS}; IDX = {p: Z[f"idx|{p}"].astype(int) for p in A.PROTOS}
VIEWS = {"E1 shipped + pair + velocity (84)": lambda p: np.c_[REF[p], PAIR[IDX[p]], VEL[IDX[p]]], "E2 shipped + velocity (68)": lambda p: np.c_[REF[p], VEL[IDX[p]]],
         "E3 shipped + pair (64)": lambda p: np.c_[REF[p], PAIR[IDX[p]]], "E4 pair + velocity + shipped event rates (52)": lambda p: np.c_[PAIR[IDX[p]], VEL[IDX[p]], REF[p][:, 32:]]}
for bn in env.EXPL:
    for p in A.PROTOS:
        Z[f"GE|{p}"] = REF[p]
    base = A.system(bn, A.FAMS)["out"]; print(SC.absline(bn + " shipped view (48)", base), flush=True)
    for name, f in VIEWS.items():
        for p in A.PROTOS:
            Z[f"GE|{p}"] = f(p)
        r = SC.compare(A.system(bn, A.FAMS)["out"], base); print("   " + SC.line(name, r)[:150], flush=True)
for p in A.PROTOS:
    Z[f"GE|{p}"] = REF[p]
