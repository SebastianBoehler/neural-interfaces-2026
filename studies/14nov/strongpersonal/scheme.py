"""s247: per-scheme view and orders > 0 for the shared (both-people) ceiling settings of ceil2.txt.
Usage: python studies/14nov/strongpersonal/scheme.py -> runs/strongpersonal/scheme.txt"""
import numpy as np
import common as C
from ceil2 import member  # noqa: E402

ER = C.routed("ER")
SET = {"B": (("s1>s2", "s1>s3"), (("PFT200", 0.25),)), "A": (("s12>s3",), (("TSp10", 0.25), ("PFT12_200", 0.5))),
       "B-oracleP10": (("s1>s2", "s1>s3"), (("TSw5", 0.25), ("PFT60", 1.0)))}
lines = []
for prot, (names, setting) in SET.items():
    for q in C.SEEN:
        d = {}
        for o, Ps in ER.items():
            v = []
            for name in names:
                te = C.rows(q, name[-1]); ms = [member(q, name, k) for k, _ in setting]
                for P in Ps:
                    Q = P
                    for M, (_, w) in zip(ms, setting):
                        Q = C.fuse(Q, M, w)
                    v.append(C.ba(te, Q) - C.ba(te, P))
            d[o] = np.mean(v)
        sc = {s: np.mean([x for o, x in d.items() if o.startswith(s)]) for s in ("within", "full", "runs")}
        lines.append(f"{prot:12s} {q} {setting}: " + " ".join(f"{s} {x:+.2f}" for s, x in sc.items())
                     + f" | orders > 0: {sum(x > 0 for x in d.values())}/16, min {min(d.values()):+.2f}")
print("\n".join(lines)); open(f"{C.OUT}/scheme.txt", "w").write("\n".join(lines) + "\n")
