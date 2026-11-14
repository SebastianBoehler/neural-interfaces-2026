"""Pseudo-labels for the pl arms of tttssl.py (ledger s255): the r15 chain's V6Fr output in order within0, mean of its six
bases -> runs/tttssl/plabels_within0.npz (proba (2655, 3), global index order). Usage: python studies/14nov/tttssl/make_pl.py"""
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[3]; z = np.load(R / "runs/chainout/s242r15_within0.npz")
P = np.mean([z[k] for k in z.files if k.endswith("|V6Fr")], 0); assert P.shape == (2655, 3) and np.isfinite(P).all()
y = np.load(R / "data/proteus_windows.npz")["y"]; (R / "runs/tttssl").mkdir(parents=True, exist_ok=True)
np.savez(R / "runs/tttssl/plabels_within0.npz", proba=P); print(f"pseudo-label accuracy {np.mean(P.argmax(1) == y):.4f}")
