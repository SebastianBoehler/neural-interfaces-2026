"""POOLCMP (ledger s191): paired comparison of two shufcombo outputs for the same order, the pooled arm (FT tag ...X) against
its retrained control (...C). Per arm (P0r = the re-streamed members without FT, V6Fr = the combination, V6r = the
fallback): six-base mean of the per-person difference, an 80 % person bootstrap, and the mean without P10.
Usage: python studies/13nov/poolcmp.py runs/shufcombo_<...>X.npz runs/shufcombo_<...>C.npz"""
import sys
import numpy as np
a, b = (np.load(f) for f in sys.argv[1:3]); bases = sorted({k.split("|")[0] for k in a.files}); assert bases == sorted({k.split("|")[0] for k in b.files})
rng = np.random.default_rng(0)
print(f"POOLCMP {sys.argv[1]} - {sys.argv[2]} ({len(bases)} bases)")
for arm, name in (("P0r", "members only (no FT, no eye/rest)"), ("V6Fr", "combination (gC5 + eye + rest)"), ("V6r", "fallback (eye + rest)")):
    d = np.mean([a[f"{bn}|{arm}"] - b[f"{bn}|{arm}"] for bn in bases], 0); assert len(d) == 10, "person count"
    bt = np.percentile([d[rng.integers(0, 10, 10)].mean() for _ in range(4000)], [10, 90])
    print(f"  {name:36s} pooled - control: new {d.mean():+.2f} [{bt[0]:+.2f}, {bt[1]:+.2f}] without P10 {np.delete(d, 9).mean():+.2f} | per person " + " ".join(f"{x:+.1f}" for x in d), flush=True)
