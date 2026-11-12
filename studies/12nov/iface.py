"""IFACE (ledger s103b): the interface as a session-stage variable (oracle labels), grid chain, new people, six bases.
Usage: python studies/12nov/iface.py"""
import itertools, os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
sys.path.insert(0, REPO + "/replay")
import env  # noqa: E402
import grid as G  # noqa: E402
p = "all_loso"; Z = G.Z; sess = Z[f"sess|{p}"]; IT = np.load(REPO + "/data/proteus_windows.npz", allow_pickle=True)["interface"].astype(str)
assert (Z[f"idx|{p}"] == np.arange(len(sess))).all(); SI = np.array([f"{s}|{i}" for s, i in zip(sess, IT)]); BH = IT == "BrainHero"
G1 = [G.PEOPLE.index(q) for q in env.A.SC.G1]; G2 = [i for i in range(10) if i not in G1]; rng = np.random.default_rng(0)


def chain(base, per_iface_rebal=False, wt=(5.0, 5.0)):
    Ps, ws = Z[f"{base}|Ps|{p}"], Z[f"{base}|ws|{p}"].copy(); ws[0] = 2.0; mix = np.zeros(Ps.shape[1:])
    for m, w in ((BH, wt[0]), (~BH, wt[1])):
        ws[-1] = w; mix[m] = sum(x / ws.sum() * P[m] for x, P in zip(ws, Ps))
    b = G.rebal(mix, SI if per_iface_rebal else sess, 1.15)
    e = G.SE.apply(Z[f"GE|{p}"], b, sess, lam=0.5, n0=60, m0=20, shrink=0.7); L, tt = G.tsem_lik(e, sess, Z[f"pc128|{p}"]); return G.scores(G.apply_lam(e, L, tt, 0.15), p)


def boot(d):
    b = [d[rng.integers(0, 10, 10)].mean() for _ in range(4000)]; return f"{d.mean():+.2f} [{np.percentile(b, 10):+.2f}, {np.percentile(b, 90):+.2f}]"


D1, D2 = [], []
for bn in env.BASES:
    ref = chain(bn); d1 = chain(bn, True) - ref; D1.append(d1)
    R = {wt: chain(bn, False, wt) for wt in itertools.product((3.0, 5.0, 7.0), repeat=2)}
    cf = np.zeros(10)
    for tr, te in ((G1, G2), (G2, G1)):
        best = max(R, key=lambda w: R[w][tr].mean()); cf[te] = R[best][te] - ref[te]
    D2.append(cf); print(f"{bn}: per-interface rebalancing {d1.mean():+.2f} | teacher weight per interface, cross-fit {cf.mean():+.2f} | best in-sample {max(v.mean() for v in R.values()) - ref.mean():+.2f}", flush=True)
print(f"MEAN six bases: per-interface rebalancing {boot(np.mean(D1, 0))} | per-interface teacher weight (cross-fit) {boot(np.mean(D2, 0))}")
