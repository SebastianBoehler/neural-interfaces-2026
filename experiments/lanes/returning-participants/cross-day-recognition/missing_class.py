"""Test 3: held-new cells (subject/session/interface) where a class has <= 2 predictions, per arm and stage (full0, within0 banks).
Usage: missing_class.py BANKDIR   (full0.npz / within0.npz of ni26_final_stack_20261120). Labels used only to name the true class counts."""
import sys
import numpy as np
d = sys.argv[1]
f = np.load("data/proteus_windows.npz", allow_pickle=True)
y, sub, ses, itf = (f[k] for k in ("y", "subject", "session", "interface"))
print("task labels:", np.unique(y))
for order in ("full0", "within0"):
    z = np.load(f"{d}/{order}.npz")
    idx = z["index"]
    cell = np.array([f"{sub[i]}|{ses[i]}|{itf[i]}" for i in idx])
    cells = sorted(set(cell))
    print(f"\n== {order}: {len(cells)} cells, {len(idx)} windows")
    for arm in ("stock", "hard", "cb", "hardcb"):
        for stage in ("raw", "filtered", "terminal"):
            P = z[f"{arm}|{stage}"].argmax(-1)
            bad, mn = [], 10**9
            for c in cells:
                m = cell == c
                n = np.bincount(P[m], minlength=3)
                mn = min(mn, n.min() / n.sum())
                if n.min() <= 2:
                    bad.append((c, n.tolist(), np.bincount(y[idx][m], minlength=3).tolist()))
            print(f"{arm:7s}{stage:9s} cells with a class <=2 preds: {len(bad)}/{len(cells)}  min class share {mn:.3f}", "" if stage != "terminal" else bad)
