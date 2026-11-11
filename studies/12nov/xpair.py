"""XPAIR (ledger s42): own-space personal stage across real session pairs of P04 and P10, on the LOSO chain.
Usage: python studies/12nov/xpair.py"""
import os, sys, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
REPO = os.path.abspath(__file__ + "/../../..")
sys.path[:0] = [REPO + "/replay", REPO + "/studies/12nov"]
import env  # noqa: E402
import ownspace as O  # noqa: E402
warnings.filterwarnings("ignore")
A = env.A; Z = A.L.Z; PC = Z["pc128|all_loso"]; assert (Z["idx|all_loso"] == np.arange(len(O.y))).all()
lg = lambda P: np.log(np.clip(P, 1e-3, 1))  # noqa: E731
zs = lambda F: (F - F.mean(0)) / (F.std(0) + 1e-9)  # noqa: E731
rows = lambda p, s: np.flatnonzero((O.PER == p) & (O.SES == s))  # noqa: E731


def members(train, te):
    o = O.Personal(train).stream(te)
    clf = LogisticRegression(C=0.1, max_iter=1000).fit(np.concatenate([zs(PC[r]) for r in train]), np.concatenate([O.y[r] for r in train]))
    o["P"] = clf.predict_proba(O.causal_z(PC[te])); return o


def stage(P, mem, keys, beta):
    L = np.log(np.clip(P, 1e-9, 1)) + beta * sum(lg(mem[k]) for k in keys) / len(keys); e = np.exp(L - L.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)


JOBS = [(p, (a,), b) for p in ("P04", "P10") for a in "123" for b in "123" if a != b] + [(p, tuple(x for x in "123" if x != b), b) for p in ("P04", "P10") for b in "123"]
MEM = {j: members([rows(j[0], a) for a in j[1]], rows(j[0], j[2])) for j in JOBS}
ARMS = [(k, b) for b in (0.25, 0.5) for k in ("P", "TCS", "PTCS")]
for bn in ("B2_pool", "B2_s345"):
    base = A.system(bn, A.FAMS)["out"]["all_loso"]; G = {}
    print(f"\n{bn}\nperson train -> test  chain | members alone P T C S | " + " ".join(f"{k}@{b}" for k, b in ARMS))
    for j in JOBS:
        r = rows(j[0], j[2]); b0 = O.ba(r, base[r]); G[j] = [O.ba(r, stage(base[r], MEM[j], k, b)) - b0 for k, b in ARMS]
        print(f"{j[0]}  s{'+'.join(j[1]):3s} -> s{j[2]}   {b0:5.1f} | " + " ".join(f"{O.ba(r, MEM[j][k]):4.0f}" for k in "PTCS") + " | " + " ".join(f"{g:+6.1f}" for g in G[j]))
    for tag, sel in (("one session, 12 pairs", [j for j in JOBS if len(j[1]) == 1]), ("   P04", [j for j in JOBS if len(j[1]) == 1 and j[0] == "P04"]),
                     ("   P10", [j for j in JOBS if len(j[1]) == 1 and j[0] == "P10"]), ("two sessions, 6", [j for j in JOBS if len(j[1]) == 2]),
                     ("   P04", [j for j in JOBS if len(j[1]) == 2 and j[0] == "P04"]), ("   P10", [j for j in JOBS if len(j[1]) == 2 and j[0] == "P10"])):
        m = np.mean([G[j] for j in sel], 0); print(f"{tag:22s} mean gain: " + " ".join(f"{k}@{b} {v:+.2f}" for (k, b), v in zip(ARMS, m)))
    one = [j for j in JOBS if len(j[1]) == 1]; print(f"pairs with PTCS@0.25 below -1: {sum(G[j][2] < -1 for j in one)} of 12")
