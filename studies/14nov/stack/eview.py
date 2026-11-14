"""STACK (ledger s242): the eye-net embedding view E_pca4 (s234, +0.14) for every order with eye nets, not only the six
s234 scored. Same builder (embview.load/fold_maps/apply: per held-out person, PCA 4 fitted on each key's own rows of the
other nine people). Checks the six existing keys reproduce runs/views_emb/E_pca4.npz exactly, then writes
runs/views_emb/E_pca4_all.npz ('rec' and '<scheme><seed>' for every runs/ocushuf/<order> with nine seeds).
Usage: python studies/14nov/stack/eview.py"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../../.."); sys.path.insert(0, REPO + "/studies/14nov/embview")
import embview as ev  # noqa: E402
orders = tuple(sorted(o for o in os.listdir(f"{REPO}/runs/ocushuf") if len([f for f in os.listdir(f"{REPO}/runs/ocushuf/{o}") if f.startswith("oc_nomix-all_loso-s")]) == 9))
ev.ORDERS = orders; X, on_rec = ev.load("E"); assert not on_rec
V = {k: ev.apply(ev.fold_maps(Xk, "pca", 4), Xk) for k, Xk in X.items()}
old = np.load(f"{REPO}/runs/views_emb/E_pca4.npz")
for k in old.files:
    err = np.abs(V[k] - old[k]).max(); print(f"{k}: max |diff| against E_pca4.npz {err:.2e}"); assert err < 1e-6, k
np.savez(f"{REPO}/runs/views_emb/E_pca4_all.npz", **V); print(f"written runs/views_emb/E_pca4_all.npz: {sorted(V)}")
