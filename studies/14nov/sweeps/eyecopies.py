"""s235: the eye view's weight as extra copies (EXTRA_VIEW file for shufcombo). EYEW scales the eye columns, but SessionEM
standardises every column of its view, so any EYEW > 0 is an exact no-op (s235 wave 1: identical outputs on six orders).
Copies are not: with the 0.7 shrinkage towards a scaled identity, k identical columns carry k times the weight.
Writes runs/diag/s235/eye<K>_<order>.npz, key <order>: K copies of the eye view exactly as shufcombo builds it for
OCU_DIR=<ocudir> (log of the clipped seed-mean posteriors, new people, recording layout).
Usage (repo root): python studies/14nov/sweeps/eyecopies.py ORDER K OCU_DIR"""
import glob, os, sys
import numpy as np

o, k, ocd = sys.argv[1], int(sys.argv[2]), sys.argv[3].rstrip("/")
fs = sorted(glob.glob(f"{ocd}/oc_nomix-all_loso-s*.npz")); assert fs, ocd
zs = [np.load(f) for f in fs]
assert all((z["index"] == np.arange(len(z["index"]))).all() for z in zs), "eye outputs not in recording layout"
ocu = np.log(np.clip(np.mean([z["proba"] for z in zs], 0), 1e-3, 1))
os.makedirs("runs/diag/s235", exist_ok=True); out = f"runs/diag/s235/eye{k}_{o}.npz"
np.savez(out, **{o: np.tile(ocu, k)}); print(out, ocu.shape, "x", k, "from", len(fs), "seed files", flush=True)
