"""SEGFP (ledger s62): the plain chain with the stage streams cut per interface, per run, or merged per person.
Usage: python studies/12nov/segfp.py"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
sys.path[:0] = [REPO + "/replay"]
import env  # noqa: E402
A = env.A; SC = A.SC; L = A.L; Z = L.Z
REF = {p: Z[f"sess|{p}"].copy() for p in A.PROTOS}


def ids(p, kind):
    s, idx, per = REF[p], Z[f"idx|{p}"].astype(int), Z[f"person|{p}"]
    key = {"session": s.astype(str), "interface": np.char.add(np.char.add(s.astype(str), "|"), L.IFACE[idx]),
           "run": np.char.add(np.char.add(s.astype(str), "|"), np.char.add(L.IFACE[idx], L.RUN[idx].astype(str))), "person": per.astype(str)}[kind]
    return np.unique(key, return_inverse=True)[1]


def scores(bn, kind):
    out = {}
    for p in A.PROTOS:
        Z[f"sess|{p}"] = ids(p, kind)
    for p in A.PROTOS:
        mix, b, e, t = L.chain(bn, p, stages=True); out[p] = (e, t)
    for p in A.PROTOS:
        Z[f"sess|{p}"] = REF[p]
    return out


for bn in env.EXPL:
    R = {k: scores(bn, k) for k in ("session", "interface", "run", "person")}
    lens = {k: int(np.median(np.unique(ids("all_loso", k), return_counts=True)[1])) for k in R}
    print(f"{bn}: median stream length " + " ".join(f"{k} {v}" for k, v in lens.items()), flush=True)
    for lvl, name in ((0, "SessionEM level"), (1, "TSEM level")):
        n0, s0 = SC.person(R["session"]["all_loso"][lvl], "all_loso"), SC.person(R["session"]["seen_session"][lvl], "seen_session")
        print(f"   {name}: per session new {n0.mean():.2f} seen {s0.mean():.2f} J {0.5 * n0.mean() + 0.5 * s0.mean():.2f}")
        for k in ("interface", "run", "person"):
            n, s = SC.person(R[k]["all_loso"][lvl], "all_loso"), SC.person(R[k]["seen_session"][lvl], "seen_session"); dn, ds = n - n0, s - s0
            multi = [SC.PEOPLE.index(q) for q in ("P04", "P10")]
            print(f"      per {k:9s}: dJ {0.5 * dn.mean() + 0.5 * ds.mean():+.2f} new {dn.mean():+.2f} seen {ds.mean():+.2f} | new people: " + " ".join(f"{q[1:]}:{v:+.1f}" for q, v in zip(SC.PEOPLE, dn))
                  + (f" | P04/P10 as new {dn[multi].mean():+.2f}, as seen {ds.mean():+.2f}" if k == "person" else ""), flush=True)
    for k in ("interface", "run"):
        st = [0.5 * (SC.person(R[k]["all_loso"][l], "all_loso") - SC.person(R["session"]["all_loso"][l], "all_loso")).mean()
              + 0.5 * (SC.person(R[k]["seen_session"][l], "seen_session") - SC.person(R["session"]["seen_session"][l], "seen_session")).mean() for l in (0, 1)]
        print(f"   per {k}: SessionEM-level {st[0]:+.2f}, TSEM-level {st[1]:+.2f}, TSEM step {st[1] - st[0]:+.2f}   (sealed after the fix: -0.72, -1.17, step -0.45)")
