"""PTYPE (ledger s67): person-to-person transfer of own-space members. Usage: python studies/12nov/ptype.py"""
import os, sys
import numpy as np
from scipy.stats import spearmanr
REPO = os.path.abspath(__file__ + "/../../..")
sys.path[:0] = [REPO + "/replay", REPO + "/studies/12nov"]
import env  # noqa: E402
import ownspace as O  # noqa: E402
A = env.A; chain = A.system("B2_pool", A.FAMS)["out"]["all_loso"]; PEOPLE = sorted(set(O.PER)); lg = lambda P: np.log(np.clip(P, 1e-3, 1))  # noqa: E731
sess = lambda p: [np.flatnonzero((O.PER == p) & (O.SES == s)) for s in sorted(set(O.SES[O.PER == p]))]  # noqa: E731
M = {p: O.Personal(sess(p)) for p in PEOPLE}; ACC = np.full((10, 10), np.nan); AGR = np.full((10, 10), np.nan)
for j, t in enumerate(PEOPLE):
    for i, s in enumerate(PEOPLE):
        a, g = [], []
        for r in sess(t):
            if s == t:
                others = [q for q in sess(t) if q[0] != r[0]]
                if not others:
                    continue
                o = O.Personal(others).stream(r)
            else:
                o = M[s].stream(r)
            L = lg(o["T"]) + lg(o["C"]) + lg(o["S"]); a.append(O.ba(r, L)); g.append(100 * np.mean(L.argmax(1) == chain[r].argmax(1)))
        if a:
            ACC[i, j], AGR[i, j] = np.mean(a), np.mean(g)
print("balanced accuracy; rows = source person, columns = target person (diagonal: other sessions of the same person)")
print("      " + " ".join(f"{p:>5s}" for p in PEOPLE))
for i, s in enumerate(PEOPLE):
    print(f"{s}   " + " ".join("    ." if np.isnan(v) else f"{v:5.0f}" for v in ACC[i]))
off = ACC.copy(); np.fill_diagonal(off, np.nan); best = np.nanmax(off, 0); pick = np.array([off[np.nanargmax(np.where(np.eye(10, dtype=bool), np.nan, AGR)[:, j]), j] for j in range(10)])
rho = [spearmanr(AGR[np.arange(10) != j, j], off[np.arange(10) != j, j])[0] for j in range(10)]
print(f"mean single source {np.nanmean(off):.1f} | best single source per target " + " ".join(f"{v:.0f}" for v in best) + f" (mean {best.mean():.1f})")
print("best source for each target: " + " ".join(f"{PEOPLE[j][1:]}<-{PEOPLE[int(np.nanargmax(off[:, j]))][1:]}" for j in range(10)))
print(f"source picked by agreement with the chain: " + " ".join(f"{v:.0f}" for v in pick) + f" (mean {pick.mean():.1f}); spearman(agreement, accuracy) per target mean {np.mean(rho):+.2f}")
print(f"symmetry: correlation of ACC[i, j] with ACC[j, i] over pairs {np.corrcoef(off[np.triu_indices(10, 1)], off.T[np.triu_indices(10, 1)])[0, 1]:+.2f}")
