"""PSEL (ledger s41): per-person choice of the personal model's space; ten-person within-session proxy on the LOSO chain.
Usage: python studies/12nov/psel.py"""
import os, sys, warnings
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
REPO = os.path.abspath(__file__ + "/../../..")
sys.path[:0] = [REPO + "/replay", REPO + "/studies/12nov"]
import env  # noqa: E402
import ownspace as O  # noqa: E402
warnings.filterwarnings("ignore")
A = env.A; SC = A.SC; Z = A.L.Z
z = np.load(REPO + "/data/proteus_windows.npz", allow_pickle=True); RUN = np.char.add(z["task"].astype(str), z["run"].astype(str))
PC = Z["pc128|all_loso"]; assert (Z["idx|all_loso"] == np.arange(len(O.y))).all()
lg = lambda P: np.log(np.clip(P, 1e-3, 1))  # noqa: E731
zs = lambda F: (F - F.mean(0)) / (F.std(0) + 1e-9)  # noqa: E731


def pc_head(tr_rows, te_rows):
    clf = LogisticRegression(C=0.1, max_iter=1000).fit(zs(PC[tr_rows]), O.y[tr_rows]); return clf.predict_proba(O.causal_z(PC[te_rows]))


def members(Arows, Brows):
    o = O.Personal([Arows]).stream(Brows); o["P"] = pc_head(Arows, Brows); return o


def gate_gap(Arows):
    """leave-one-run-out inside the public part: accuracy of T.C.S minus accuracy of P (percentage points)."""
    runs = RUN[Arows]; own, pop = np.zeros((len(Arows), 3)), np.zeros((len(Arows), 3))
    for u in dict.fromkeys(runs):
        te, tr = runs == u, runs != u; o = members(Arows[tr], Arows[te]); own[te] = lg(o["T"]) + lg(o["C"]) + lg(o["S"]); pop[te] = o["P"]
    y = O.y[Arows]; return 100 * (np.mean(own.argmax(1) == y) - np.mean(pop.argmax(1) == y))


SETS = {"P": "P", "TCS": "TCS", "PTCS": "PTCS"}
PEOPLE = sorted(set(O.PER)); M, GAP, ROWS = {}, {}, {}
for p in PEOPLE:
    s1 = np.flatnonzero((O.PER == p) & (O.SES == "1")); order = list(dict.fromkeys(RUN[s1]))
    Ar, Br = s1[np.isin(RUN[s1], order[:3])], s1[np.isin(RUN[s1], order[3:])]
    M[p] = members(Ar, Br); GAP[p] = gate_gap(Ar); ROWS[p] = Br
    print(f"{p}: public part {len(Ar)} windows, later part {len(Br)}; gate gap {GAP[p]:+.1f}; members alone on the later part: "
          + " ".join(f"{k} {O.ba(Br, M[p][k]):.1f}" for k in "PTCS"), flush=True)


def stage(P, mem, keys, beta):
    L = np.log(np.clip(P, 1e-9, 1)) + beta * sum(lg(mem[k]) for k in keys) / len(keys); e = np.exp(L - L.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)


for bn in ("B2_pool", "B2_s345"):
    base = A.system(bn, A.FAMS)["out"]["all_loso"]; b0 = np.array([O.ba(ROWS[p], base[ROWS[p]]) for p in PEOPLE])
    print(f"\n{bn}: LOSO chain on the later part, mean {b0.mean():.2f} | " + " ".join(f"{p[1:]}:{v:.0f}" for p, v in zip(PEOPLE, b0)))
    G = {}
    for beta in (0.25, 0.5):
        for name, keys in SETS.items():
            g = np.array([O.ba(ROWS[p], stage(base[ROWS[p]], M[p], keys, beta)) for p in PEOPLE]) - b0; G[(name, beta)] = g
            print(f"   {name:5s} beta {beta}: mean {g.mean():+.2f} | " + " ".join(f"{p[1:]}:{v:+.1f}" for p, v in zip(PEOPLE, g)) + f" | hurt by > 1: {(g < -1).sum()}")
        pick = ["TCS" if GAP[p] >= 5 else "P" if GAP[p] <= -5 else "PTCS" for p in PEOPLE]
        sel = np.array([G[(k, beta)][i] for i, k in enumerate(pick)]); best = max(G[(k, beta)].mean() for k in SETS)
        rho = spearmanr([GAP[p] for p in PEOPLE], G[("TCS", beta)] - G[("P", beta)])[0]
        print(f"   gate  beta {beta}: mean {sel.mean():+.2f} (best uniform {best:+.2f}; picks {' '.join(pick)}) | spearman(gap, TCS - P) {rho:+.2f}")
