"""SEEN-DOSE (ledger s111), proxy part: forms of the own-space stage on the ten-person in-population proxy (runs 1-3
public, 4-6 later; in-population nets as §82). Usage: python studies/12nov/seendose.py"""
import itertools, os, sys
import numpy as np
F = os.path.abspath(__file__ + "/../inpop.py"); exec(compile(open(F).read().split("\nMEM = {}")[0], F, "exec"))  # noqa: S102
lg = lambda P: np.log(np.clip(P, 1e-3, 1))  # noqa: E731
STREAMS = {}
for kref in (1.0, 2.0, 4.0):
    O.KREF = kref
    for q in PEOPLE:
        a = np.flatnonzero((O.PER == q) & (O.SES == "1") & (RUN <= 3)); r = B[O.PER[B] == q]; STREAMS[kref, q] = O.Personal([a]).stream(r)
O.KREF = 2.0; LATE = {q: B[O.PER[B] == q] for q in PEOPLE}
ba = lambda P: np.array([O.ba(LATE[q], P[O.PER[B] == q]) for q in PEOPLE])  # noqa: E731
FORMS = list(itertools.product((0.05, 0.10, 0.15, 0.25, 0.40), (1.0, 2.0, 4.0), ("TCS", "TC", "C", "T", "S")))
R = {}
for bn in ("B2_pool", "B2_s345", "B2_s012"):
    P0 = chain(bn, True); r0 = ba(P0)
    for beta, kref, keys in FORMS:
        S = P0.copy()
        for q in PEOPLE:
            k = O.PER[B] == q; m = STREAMS[kref, q]; L = np.log(np.clip(P0[k], 1e-9, 1)) + beta * sum(lg(m[x]) for x in keys) / len(keys)
            e = np.exp(L - L.max(1, keepdims=True)); S[k] = e / e.sum(1, keepdims=True)
        R.setdefault((beta, kref, keys), []).append(ba(S) - r0)
ref = np.mean(R[0.10, 2.0, "TCS"], 0)
print(f"v5.1 form (beta 0.10, KREF 2, TCS) on the proxy: {ref.mean():+.2f} | " + " ".join(f"{x:+.1f}" for x in ref))
top = sorted(FORMS, key=lambda f: -np.mean(R[f]))[:12]
for f in top:
    d = np.mean(R[f], 0); print(f"  beta {f[0]:.2f} KREF {f[1]:.0f} {f[2]:3s}: {d.mean():+.2f} (against v5.1's form {d.mean() - ref.mean():+.2f}; ahead for {(d > ref).sum()}/10) | " + " ".join(f"{x:+.1f}" for x in d))
