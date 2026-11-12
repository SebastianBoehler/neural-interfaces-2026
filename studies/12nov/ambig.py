"""AMBIG (ledger s88): own-space members with true labels, run-wise within each session, on the chain's hard windows.
Usage: python studies/12nov/ambig.py"""
import os, sys
import numpy as np
from joblib import Parallel, delayed
REPO = os.path.abspath(__file__ + "/../../..")
sys.path[:0] = [REPO + "/replay", REPO + "/studies/12nov"]
import env  # noqa: E402
import ownspace as O  # noqa: E402
A = env.A; Z = A.L.Z; p = "all_loso"; sess = Z[f"sess|{p}"]; y = O.y.astype(int); RUN = np.load(REPO + "/data/proteus_windows.npz", allow_pickle=True)["run"].astype(int)


def own(r):
    out = np.zeros((len(r), 3))
    for k in sorted(set(RUN[r])):
        te, tr = r[RUN[r] == k], r[RUN[r] != k]; o = O.Personal([tr]).stream(te); out[RUN[r] == k] = sum(np.log(np.clip(o[x], 1e-3, 1)) for x in "TCS")
    return out


S = list(np.unique(sess)); OWN = np.zeros((len(y), 3))
for s, o in zip(S, Parallel(n_jobs=6)(delayed(own)(np.flatnonzero(sess == s)) for s in S)):
    OWN[sess == s] = o
C = A.system("B2_pool", A.FAMS)["out"][p]; cf = C.max(1); h = np.zeros(len(y), bool)
for s in S:
    r = np.flatnonzero(sess == s); h[r[cf[r] <= np.quantile(cf[r], 1 / 3)]] = True
po, pc = OWN.argmax(1), C.argmax(1); wr = h & (pc != y)
print(f"own model (true labels, other runs): all {100 * np.mean(po == y):.1f}% | chain-hard {100 * np.mean(po[h] == y[h]):.1f}% | rest {100 * np.mean(po[~h] == y[~h]):.1f}%")
print(f"chain: hard {100 * np.mean(pc[h] == y[h]):.1f}% | rest {100 * np.mean(pc[~h] == y[~h]):.1f}%")
print(f"hard windows where the chain is wrong ({wr.sum()}): own model picks the true class {100 * np.mean(po[wr] == y[wr]):.1f}%, the chain's class {100 * np.mean(po[wr] == pc[wr]):.1f}%")
for q in sorted(set(O.PER)):
    m = O.PER == q; print(f"  {q}: own hard {100 * np.mean(po[m & h] == y[m & h]):.0f} rest {100 * np.mean(po[m & ~h] == y[m & ~h]):.0f} | chain hard {100 * np.mean(pc[m & h] == y[m & h]):.0f} rest {100 * np.mean(pc[m & ~h] == y[m & ~h]):.0f} | own right where chain wrong {100 * np.mean(po[m & wr] == y[m & wr]):.0f}")
