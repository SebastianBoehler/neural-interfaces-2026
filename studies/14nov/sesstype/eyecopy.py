"""s236: a real eye-view weight for SessionEM. EMg z-scores every view column by the session's own sd (replay/gm.py
EMg._adapt), so shufcombo's EYEW (column scaling, s231) is a no-op for every w > 0 and only w = 0 changes the output.
k extra copies of the eye view, appended through EXTRA_VIEW, raise its share of the shrunk Mahalanobis distance instead.
Writes runs/sesstype/eyecopy_k<k>.npz: '<scheme><seed>' -> (N, 3k), recording layout, the same columns as shufcombo's
eye view (log of the clipped seed-mean ocushuf posterior of that order; shufcombo indexes XV[P] like OCU[P]). No 'rec'
key, so an order without its own rows fails instead of falling back. Usage: python studies/14nov/sesstype/eyecopy.py"""
import glob, os
import numpy as np
REPO = os.path.abspath(__file__ + "/../../../..")
ORDERS = os.environ.get("ORDERS", "within0,within1,full0,full1,runs0,runs1").split(",")
os.makedirs(REPO + "/runs/sesstype", exist_ok=True)
for k in (1, 2):
    out = {}
    for o in ORDERS:
        fs = sorted(glob.glob(f"{REPO}/runs/ocushuf/{o}/oc_nomix-all_loso-s*.npz")); assert fs, o
        ocu = np.log(np.clip(np.mean([np.load(f)["proba"] for f in fs], 0), 1e-3, 1)); out[o] = np.tile(ocu, (1, k))
    np.savez(f"{REPO}/runs/sesstype/eyecopy_k{k}.npz", **out); print(f"eyecopy_k{k}: {out[ORDERS[0]].shape} x {len(out)} orders")
