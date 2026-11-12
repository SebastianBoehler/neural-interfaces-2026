"""Inputs for S3 (ledger s98): S2's standardised multi-view features per fold, from runs/sys2_feats.npz.
Usage: python studies/12nov/sys3_inputs.py  -> runs/sys3_inputs.npz"""
import os, sys
import numpy as np
from sklearn.decomposition import PCA
REPO = os.path.abspath(__file__ + "/../../..")
sys.path[:0] = [REPO + "/replay", REPO + "/studies/12nov"]
import env  # noqa: E402
import ownspace as O  # noqa: E402
Z = env.A.L.Z; FF = np.load(REPO + "/runs/sys2_feats.npz"); out = {}
for q in sorted(set(O.PER)):
    tr = O.PER != q; T = FF[f"T_{q}"].astype(np.float64); T = (T - T[tr].mean(0)) / (T[tr].std(0) + 1e-6)
    X = np.c_[PCA(256, random_state=0).fit(T[tr]).transform(T), PCA(64, random_state=0).fit(FF["SL"][tr]).transform(FF["SL"]), FF["BP"], FF["GE"]]
    X = np.clip((X - X[tr].mean(0)) / (X[tr].std(0) + 1e-6), -6, 6); out[f"X_{q}"] = X.astype(np.float32); print(q, X.shape, flush=True)
np.savez(REPO + "/runs/sys3_inputs.npz", y=O.y, person=O.PER, sess=Z["sess|all_loso"].astype(str), **out)
