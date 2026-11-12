"""NETVIEW2 (ledger s73): second TSEM through a net's classifier-input view (stored by netemb.py), after the full routed
v4.1 replay chain. Usage: python studies/12nov/netview2.py ARCH SEEDS [lams=0.15] [proto=all_loso]   e.g. fb 2,3,4,5"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
sys.path.insert(0, REPO + "/replay")
import env  # noqa: E402
import grid as G  # noqa: E402
A = env.A; arch, seeds = sys.argv[1], [int(v) for v in sys.argv[2].split(",")]; kw = dict(a.split("=") for a in sys.argv[3:])
LAMS = [float(v) for v in kw.get("lams", "0.15").split(",")]; p = kw.get("proto", "all_loso"); rng = np.random.default_rng(0)
Z = A.L.Z; sess = Z[f"sess|{p}"]; idx = Z[f"idx|{p}"].astype(int)


def view(seed):
    z = np.load(f"{REPO}/runs/netemb/ne_{arch}-{p}-s{seed}.npz"); pos = {int(g): k for k, g in enumerate(z["index"])}
    E = z["emb"].astype(np.float64)[[pos[int(g)] for g in idx]]; keep = E.std(0) > 1e-6; return E[:, keep]


def boot(d):
    if len(d) < 3:
        return " ".join(f"{v:+.1f}" for v in d)
    b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]; return f"{d.mean():+.2f} [{np.percentile(b, 10):+.2f}, {np.percentile(b, 90):+.2f}]"


V = {s: view(s) for s in seeds}; print(f"{arch} seeds {seeds} ({p}): view dims " + " ".join(str(v.shape[1]) for v in V.values()), flush=True)
for bn in env.EXPL:
    base = A.system(bn, A.FAMS)["out"][p]; ref = G.scores(base, p); print(f"{bn}: routed chain {ref.mean():.2f}", flush=True)
    for lam in LAMS:
        D = []
        for s in seeds:
            L, tt = G.tsem_lik(base, sess, V[s]); d = G.scores(G.apply_lam(base, L, tt, lam), p) - ref; D.append(d)
            print(f"  seed {s}, lambda {lam}: {boot(d)} | " + " ".join(f"{v:+.1f}" for v in d), flush=True)
        m = np.mean(D, 0); print(f"  MEAN over seeds, lambda {lam}: {boot(m)} | people >= 0: {(m >= 0).sum()}/{len(m)} | " + " ".join(f"{v:+.1f}" for v in m), flush=True)
