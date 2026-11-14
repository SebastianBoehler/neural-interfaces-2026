"""s247 ceiling, joint form: ER x TSp^a x TSw5^b x PFT^c with (a, b, c) on a grid and the TSp (C 0.1 / 10) and PFT
(60 / 200 steps) variants as choices. One setting per person and protocol, shared by the protocol's test sessions,
picked on those sessions (oracle) = the ceiling; plus the best single setting shared by both people (less oracle).
TSwpl of ceil.py is left out: its labels are the mean ER posterior over all 96 runs, an ensemble no package has.
Reads runs/strongpersonal/ceil_members.npz (ceil.py) and pft12 nets (protocol A) if fetched.
Usage: python studies/14nov/strongpersonal/ceil2.py -> runs/strongpersonal/ceil2.txt"""
import itertools
import numpy as np
import common as C

ER = C.routed("ER"); Mz = np.load(f"{C.OUT}/ceil_members.npz"); W = (0.0, 0.25, 0.5, 1.0)
P12 = {v: C.pft12(v) for v in ("pf60", "pf200")}
PROTS = {"B": ("s1>s2", "s1>s3"), "A": ("s12>s3",), "A-like": ("s1>s3",)}


def member(q, name, k):
    te = C.rows(q, name[-1]); P = np.full((C.N, 3), 1 / 3)
    if k.startswith("PFT12"):
        G, n = P12["pf" + k[6:]]; assert n >= 10, f"pft12 {k}: {n} nets"; P[te] = G[te]
    else:
        P[te] = Mz[f"{q}|{name}|{k}"]
    return P


def score(q, names, setting):
    """mean over the protocol's sessions of the scheme-mean BA of ER x members^weights, and of ER alone."""
    out, ref = [], []
    for name in names:
        te = C.rows(q, name[-1]); ms = [(member(q, name, k), w) for k, w in setting if w > 0]
        v, r = {}, {}
        for o, Ps in ER.items():
            sc = o.rstrip("0123456789"); f, e = [], []
            for P in Ps:
                Q = P
                for M, w in ms:
                    Q = C.fuse(Q, M, w)
                f.append(C.ba(te, Q)); e.append(C.ba(te, P))
            v.setdefault(sc, []).append(np.mean(f)); r.setdefault(sc, []).append(np.mean(e))
        out.append(np.mean([np.mean(x) for x in v.values()])); ref.append(np.mean([np.mean(x) for x in r.values()]))
    return np.mean(out), np.mean(ref)


def main():
    lines = []
    for prot, names in PROTS.items():
        pft = ("PFT60", "PFT200") if prot != "A" else (("PFT12_60", "PFT12_200") if P12["pf60"][1] else ())
        grid = []
        for tsp, pv in itertools.product(("TSp0.1", "TSp10"), pft or (None,)):
            for a, b, c in itertools.product(W, W, W if pv else (0.0,)):
                grid.append(((tsp, a), ("TSw5", b)) + (((pv, c),) if pv else ()))
        R = {q: {g: score(q, names, g) for g in grid} for q in C.SEEN}
        for q in C.SEEN:
            g = max(R[q], key=lambda g: R[q][g][0]); f, e = R[q][g]
            lines.append(f"{prot:6s} {q}: ER {e:.2f} | oracle {f:.2f} ({f - e:+.2f}) with {[(k, w) for k, w in g if w > 0]}")
        g = max(grid, key=lambda g: sum(R[q][g][0] - R[q][g][1] for q in C.SEEN))
        lines.append(f"{prot:6s} one setting for both {[(k, w) for k, w in g if w > 0]}: " + ", ".join(f"{q} {R[q][g][0] - R[q][g][1]:+.2f}" for q in C.SEEN)
                     + ("" if pft else "  [no pft12 nets: TS members only]"))
        print("\n".join(lines[-3:]), flush=True)
    open(f"{C.OUT}/ceil2.txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
