"""STACK stk2 view (ledger s242): EXTRA_VIEW for one order = two extra copies of the eye view (eye weight 3, the only
positive in s235: +0.09, full +0.16, runs +0.12, within -0.02), built exactly as studies/14nov/sweeps/eyecopies.py
(log of the clipped seed-mean eye posterior from OCU_DIR, recording layout), followed by the E_pca4 columns of that order
from runs/views_emb/E_pca4_all.npz. Writes runs/views_emb/stk2_<order>.npz, key <order>, (2655, 2 x 3 + 4).
Usage (repo root): python studies/14nov/stack/stk2view.py ORDER OCU_DIR"""
import glob, sys
import numpy as np
o, ocd = sys.argv[1], sys.argv[2].rstrip("/")
fs = sorted(glob.glob(f"{ocd}/oc_nomix-all_loso-s*.npz")); assert len(fs) == 9, (ocd, len(fs))
zs = [np.load(f) for f in fs]; assert all((z["index"] == np.arange(len(z["index"]))).all() for z in zs), "eye outputs not in recording layout"
ocu = np.log(np.clip(np.mean([z["proba"] for z in zs], 0), 1e-3, 1)); E = np.load("runs/views_emb/E_pca4_all.npz")[o]
V = np.c_[np.tile(ocu, 2), E]; assert V.shape == (len(ocu), 10), V.shape
np.savez(f"runs/views_emb/stk2_{o}.npz", **{o: V}); print(f"runs/views_emb/stk2_{o}.npz {V.shape}", flush=True)
