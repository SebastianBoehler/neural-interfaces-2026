"""LAMS (ledger s75): TSEM strength in the full routed v4.1 replay, six bases, new and seen; cross-fit over people halves.
Usage: python studies/12nov/lams.py"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
sys.path.insert(0, REPO + "/replay")
import env  # noqa: E402
A = env.A; SC = A.SC; LAMS = (0.12, 0.15, 0.20, 0.25, 0.30); G1 = [SC.PEOPLE.index(q) for q in SC.G1]; G2 = [SC.PEOPLE.index(q) for q in SC.G2]
BASES = ["B2_pool", "B2_s345", "B2_s012", "B2_s0", "B2_s1", "B2_s2"]; rng = np.random.default_rng(0); R = {}
for bn in BASES:
    for lam in LAMS:
        A.LAM = lam; s = A.summary(A.system(bn, A.FAMS)); R[bn, lam] = (np.asarray(s["dn"], float), np.asarray(s["ds"], float))
    A.LAM = 0.15
    n0, s0 = R[bn, 0.15]
    print(bn + ": " + " | ".join(f"lam {lam}: new {R[bn, lam][0].mean() - n0.mean():+.2f} seen {R[bn, lam][1].mean() - s0.mean():+.2f} (P04 {R[bn, lam][1][0] - s0[0]:+.1f})" for lam in LAMS), flush=True)
for lam in LAMS:
    d = np.mean([R[b, lam][0] - R[b, 0.15][0] for b in BASES], 0); ds = np.mean([R[b, lam][1].mean() - R[b, 0.15][1].mean() for b in BASES])
    bt = np.percentile([d[rng.integers(0, 10, 10)].mean() for _ in range(4000)], [10, 90])
    print(f"MEAN of six bases, lam {lam}: new {d.mean():+.2f} [{bt[0]:+.2f}, {bt[1]:+.2f}] people >= 0 {(d >= 0).sum()}/10 | seen {ds:+.2f} | " + " ".join(f"{v:+.1f}" for v in d))
cf = []
for b in BASES:
    for tr, te in ((G1, G2), (G2, G1)):
        best = max(LAMS, key=lambda l: R[b, l][0][tr].mean()); cf.append(R[b, best][0][te].mean() - R[b, 0.15][0][te].mean())
print(f"cross-fit over halves, mean of 12: {np.mean(cf):+.2f} (range {min(cf):+.2f} to {max(cf):+.2f})")
