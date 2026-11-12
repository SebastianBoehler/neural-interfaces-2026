"""Nulls for NETVIEW2 (ledger s73 addendum): random 64-direction views of core1's 256 components as a second TSEM, and the
first TSEM at a higher strength, after the routed chain (new people). Usage: python studies/12nov/viewnull.py"""
import glob, os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
sys.path.insert(0, REPO + "/replay")
import env  # noqa: E402
import grid as G  # noqa: E402
A = env.A; p = "all_loso"; Z = A.L.Z; sess = Z[f"sess|{p}"]; N = len(sess); rng = np.random.default_rng(1)
PC = np.zeros((N, 256))
for f in sorted(glob.glob(f"{REPO}/runs/replay_data/feats_exact/all_loso-P*.npz")):
    z = np.load(f, allow_pickle=True); sd = z["pc_tr"].std(0)
    for k in [x for x in z.files if x.endswith("|pc")]:
        PC[z[k.replace("|pc", "|idx")].astype(int)] = z[k] / sd
assert np.abs(PC).sum(1).min() > 0


def boot(d):
    b = [d[rng.integers(0, 10, 10)].mean() for _ in range(4000)]; return f"{d.mean():+.2f} [{np.percentile(b, 10):+.2f}, {np.percentile(b, 90):+.2f}]"


for bn in env.BASES:
    sysd = A.system(bn, A.FAMS); base = sysd["out"][p]; ref = G.scores(base, p); print(f"{bn}: routed chain {ref.mean():.2f}", flush=True)
    D = []
    for draw in range(5):
        W = np.linalg.qr(rng.standard_normal((256, 64)))[0]; L, tt = G.tsem_lik(base, sess, PC @ W); d = G.scores(G.apply_lam(base, L, tt, 0.15), p) - ref; D.append(d)
        print(f"  random 64 directions, draw {draw}, lambda 0.15: {boot(d)}", flush=True)
    print(f"  random 64, MEAN of 5 draws: {boot(np.mean(D, 0))}", flush=True)
    for lam in (0.20, 0.25):
        A.LAM = lam; out = A.system(bn, A.FAMS)["out"][p]; d = G.scores(out, p) - ref; print(f"  first TSEM at lambda {lam} (routed chain rebuilt): {boot(d)}", flush=True)
    A.LAM = 0.15
