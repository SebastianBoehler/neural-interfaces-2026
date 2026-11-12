"""Score S3 (ledger s98): assemble the folds, mean of seeds, alone and as a product with the chain on six bases.
Usage: python studies/12nov/sys3_score.py runs/sys3"""
import glob, os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
sys.path.insert(0, REPO + "/replay")
import env  # noqa: E402
import grid as G  # noqa: E402
A = env.A; p = "all_loso"; N = len(G.Z[f"y|{p}"]); D = sys.argv[1]; P = np.zeros((N, 3)); cnt = np.zeros(N); rng = np.random.default_rng(0)
for f in sorted(glob.glob(f"{D}/sys3_P*-s*.npz")):
    z = np.load(f); P[z["index"]] += z["proba"]; cnt[z["index"]] += 1
assert (cnt > 0).all(), f"{(cnt == 0).sum()} rows without an S3 output"
P /= cnt[:, None]; a = G.scores(P, p); print(f"S3 alone (seeds per row {cnt.min():.0f}-{cnt.max():.0f}): {a.mean():.2f} | " + " ".join(f"{x:.0f}" for x in a))
for bn in env.BASES:
    C = A.system(bn, A.FAMS)["out"][p]; ref = G.scores(C, p)
    for w in (0.25, 0.5):
        L = np.log(np.clip(C, 1e-9, 1)) + w * np.log(np.clip(P, 1e-3, 1)); e = np.exp(L - L.max(1, keepdims=True)); d = G.scores(e / e.sum(1, keepdims=True), p) - ref
        bt = np.percentile([d[rng.integers(0, 10, 10)].mean() for _ in range(4000)], [10, 90]); print(f"  {bn}: product ^{w}: {d.mean():+.2f} [{bt[0]:+.2f}, {bt[1]:+.2f}] people >= 0 {(d >= 0).sum()}/10")
