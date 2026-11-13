import numpy as np
FE = "runs/replay_data/feats_exact/"
def load(f):
    z = np.load(FE + f, allow_pickle=True); return z["idx_tr"].astype(int), z["pc_tr"][:, :128].astype(np.float64)
for a, b in (("all_loso-P01.npz", "all_loso-P02.npz"), ("all_loso-P01.npz", "all_loso-P04.npz"), ("all_loso-P04.npz", "all_loso-P10.npz"), ("all_loso-P03.npz", "all_loso-P09.npz")):
    ia, Xa = load(a); ib, Xb = load(b); common = np.intersect1d(ia, ib)
    pa = {int(g): k for k, g in enumerate(ia)}; pb = {int(g): k for k, g in enumerate(ib)}
    A = Xa[[pa[int(g)] for g in common]]; B = Xb[[pb[int(g)] for g in common]]
    c = np.array([abs(np.corrcoef(A[:, j], B[:, j])[0, 1]) for j in range(128)])
    sg = np.array([np.sign(np.corrcoef(A[:, j], B[:, j])[0, 1]) for j in range(128)])
    print(a[9:12], b[9:12], "shared rows", len(common), "| |corr| of same-numbered component: first 8", np.round(c[:8], 2), "| median cols 0-15", round(float(np.median(c[:16])), 2), "16-63", round(float(np.median(c[16:64])), 2), "64-127", round(float(np.median(c[64:])), 2),
          "| cols with |corr| < 0.5:", int((c < 0.5).sum()), "of 128 | sign flips:", int((sg < 0).sum()))
