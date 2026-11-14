"""s240 gate 1: share of v6f's emitted errors (runs/chainout/s215_<order>_off.npz, '<base>|V6Fr', recording layout) that are
SUB <-> WORD confusions, new people (all_loso rows), pooled over six bases; by order and by interface x run type.
Usage: python studies/14nov/eyeevents/confusion.py"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../../..")
sys.path.insert(0, REPO + "/replay")
import env  # noqa: E402
A = env.A; Z = A.L.Z; p0 = "all_loso"
IDX = A.pl.B[f"idx|{p0}"].astype(int); y = Z[f"y|{p0}"].astype(int)
W = np.load(REPO + "/data/proteus_windows.npz"); assert (W["y"][IDX] == y).all()
cell = np.char.add(W["interface"][IDX].astype(str), np.char.add("|", np.where(np.char.startswith(W["task"][IDX].astype(str), "Online"), "online", "acq")))
ORDERS = [f"{s}{k}" for s in ("within", "full", "runs") for k in range(2)]
BASES = ("B2_pool", "B2_s345", "B2_s012", "B2_s0", "B2_s1", "B2_s2")
NM = ("MI", "SUB", "WORD")
tot = np.zeros((3, 3)); bycell = {c: np.zeros((3, 3)) for c in np.unique(cell)}
for o in ORDERS:
    z = np.load(f"{REPO}/runs/chainout/s215_{o}_off.npz"); C = np.zeros((3, 3))
    for b in BASES:
        pr = z[f"{b}|V6Fr"].argmax(1); np.add.at(C, (y, pr), 1)
        for c in bycell:
            m = cell == c; np.add.at(bycell[c], (y[m], pr[m]), 1)
    err = C.sum() - np.trace(C); sw = C[1, 2] + C[2, 1]
    print(f"{o:8s} acc {100 * np.trace(C) / C.sum():.1f}  SUB<->WORD {100 * sw / err:.1f} % of errors  (MI<->SUB {100 * (C[0, 1] + C[1, 0]) / err:.1f}, MI<->WORD {100 * (C[0, 2] + C[2, 0]) / err:.1f})", flush=True)
    tot += C
err = tot.sum() - np.trace(tot)
print("pooled confusion (rows true, cols predicted; 6 orders x 6 bases):"); [print(f"  {NM[i]:5s}", tot[i].astype(int)) for i in range(3)]
print(f"POOLED SUB<->WORD share of errors {100 * (tot[1, 2] + tot[2, 1]) / err:.1f} % (gate 40 %): {'PASS' if (tot[1, 2] + tot[2, 1]) / err >= 0.4 else 'STOP'}")
for c, C in bycell.items():
    e = C.sum() - np.trace(C); print(f"  {c:14s} acc {100 * np.trace(C) / C.sum():.1f}  SUB<->WORD {100 * (C[1, 2] + C[2, 1]) / e:.1f} %  MI<->SUB {100 * (C[0, 1] + C[1, 0]) / e:.1f}  MI<->WORD {100 * (C[0, 2] + C[2, 0]) / e:.1f}")
