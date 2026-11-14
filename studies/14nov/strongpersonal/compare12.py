"""s247a: what the routed path gets from a second public session, and what the strong path adds on top (session 3).
Levels: ER / T with session 1 public (s245 = chain12 S2=0) and with sessions 1 + 2 (chain12 S2=1). Strong path on top of
ER(1+2) and T(1+2): BASE x TSp^a x TSw5^b x PFT^c, the §247 grid, members from window 0 (ceiling form); PFT = the 15
pftshuf.py A nets (sessions 1 + 2 public) in each delivery order. Oracle per person, and one setting for both.
Usage: python studies/14nov/strongpersonal/compare12.py -> runs/strongpersonal/compare12.txt"""
import glob, itertools
import numpy as np
import common as C

Z0, Z1 = (np.load(f"{C.OUT}/chain12_s2{k}.npz") for k in (0, 1)); Mz = np.load(f"{C.OUT}/ceil_members.npz")
ORD = sorted({k.split("|")[0] for k in Z1.files}); BASES = sorted({k.split("|")[1] for k in Z1.files}); W = (0.0, 0.25, 0.5, 1.0)
TE = {q: C.rows(q, "3") for q in C.SEEN}; POS = {int(g): k for k, g in enumerate(C.SS)}; TSS = {q: np.array([POS[int(g)] for g in TE[q]]) for q in C.SEEN}


def pft(v):
    fs = sorted(glob.glob(f"{C.OUT}/pftshuf/A_*_{v}-s*.npz")); assert len(fs) == 15, f"A {v}: {len(fs)}"; out = {}
    for o in ORD:
        L = []
        for f in fs:
            z = np.load(f); P = np.full((C.N, 3), 1 / 3); P[z["index"].astype(int)] = z[o]; L.append(np.log(np.clip(P, 1e-4, 1)))
        G = np.exp(np.mean(L, 0)); out[o] = G / G.sum(1, keepdims=True)
    return out


MEM = {f"PFT{v[2:]}": pft(v) for v in ("pf60", "pf200")}
for k in ("TSp0.1", "TSp10", "TSw5"):
    P = {}
    for q in C.SEEN:
        P[q] = Mz[f"{q}|s12>s3|{k}"]
    MEM[k] = P


def m_at(k, q, o):  # member posterior on q's session 3 rows
    return MEM[k][o][TE[q]] if k.startswith("PFT") else MEM[k][q]


def bac(q, P3):  # P3: q's session-3 rows
    F = np.full((C.N, 3), 1 / 3); F[TE[q]] = P3; return C.ba(TE[q], F)


def scheme_mean(d):
    return np.mean([np.mean([x for o, x in d.items() if o.startswith(s)]) for s in ("within", "full", "runs")])


def level(Zf, arm, q):
    return scheme_mean({o: np.mean([bac(q, Zf[f"{o}|{bn}|{arm}"][TSS[q]].astype(float)) for bn in BASES]) for o in ORD})


lines = ["session 3, mean of schemes (16 orders x 6 bases): ER(1) / T(1) / ER(1+2) / T(1+2)"]
LV = {}
for q in C.SEEN:
    LV[q] = [level(Z0, "ER", q), level(Z0, "T", q), level(Z1, "ER", q), level(Z1, "T", q)]
    lines.append(f"  {q}: " + " / ".join(f"{x:.2f}" for x in LV[q]) + f" | second session for the routed path: ER {LV[q][2] - LV[q][0]:+.2f}, T {LV[q][3] - LV[q][1]:+.2f}")
print("\n".join(lines), flush=True)
grid = [((t, a), ("TSw5", b), (pv, c)) for t, pv in itertools.product(("TSp0.1", "TSp10"), ("PFT60", "PFT200")) for a, b, c in itertools.product(W, W, W)]
for arm, li in (("ER", 2), ("T", 3)):
    R = {q: {} for q in C.SEEN}
    for q in C.SEEN:
        for g in grid:
            d = {}
            for o in ORD:
                v = []
                for bn in BASES:
                    L = np.log(np.clip(Z1[f"{o}|{bn}|{arm}"][TSS[q]].astype(float), 1e-6, 1))
                    for k, w in g:
                        if w > 0:
                            L = L + w * np.log(np.clip(m_at(k, q, o), 1e-6, 1))
                    v.append(bac(q, np.exp(L - L.max(1, keepdims=True))))
                d[o] = np.mean(v)
            R[q][g] = scheme_mean(d) - LV[q][li]
    for q in C.SEEN:
        g = max(R[q], key=R[q].get); lines.append(f"strong on {arm}(1+2) {q}: oracle {R[q][g]:+.2f} with {[(k, w) for k, w in g if w > 0]}")
    g = max(grid, key=lambda g: sum(R[q][g] for q in C.SEEN))
    lines.append(f"strong on {arm}(1+2), one setting for both {[(k, w) for k, w in g if w > 0]}: " + ", ".join(f"{q} {R[q][g]:+.2f}" for q in C.SEEN)
                 + f"; sealed {0.2 * R['P04'][g] + 0.8 * R['P10'][g]:+.2f}")
    print("\n".join(lines[-3:]), flush=True)
open(f"{C.OUT}/compare12.txt", "w").write("\n".join(lines) + "\n")
