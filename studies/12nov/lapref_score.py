"""LAPREF (ledger s99): core1 variants from experiments/ts_core2.py swapped into core1's slot of the routed chain.
Usage: python studies/12nov/lapref_score.py runs/lapref NAME [NAME...]   (the shipped car_ov17_fs2_t0.1_le_k2 is checked first)"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
sys.path.insert(0, REPO + "/replay")
import env  # noqa: E402
A = env.A; SC = A.SC; D = sys.argv[1]; IDX = {p: A.pl.B[f"idx|{p}"].astype(int) for p in A.PROTOS}; rng = np.random.default_rng(0)


def load(name):
    z = np.load(f"{D}/{name}.npz"); out = {}
    for p in A.PROTOS:
        pos = {int(g): k for k, g in enumerate(z[f"index|{p}"])}; out[p] = z[f"lr10|{p}"][[pos[int(g)] for g in IDX[p]]].astype(np.float64)
    return out


ref_name = "car_ov17_fs2_t0.1_le_k2"
if os.path.exists(f"{D}/{ref_name}.npz"):
    R = load(ref_name); print("harness: rebuilt shipped core1 vs stored, max |diff| " + " ".join(f"{p} {np.abs(R[p] - A.C1[p]).max():.2e}" for p in A.PROTOS))
fams = [f for f in A.FAMS if f != "core1"]
for name in sys.argv[2:]:
    A.EXTRA["c1x"] = load(name); a = SC.person(A.EXTRA["c1x"]["all_loso"], "all_loso"); a = 100 * a if a.max() <= 1 else a
    print(f"{name}: alone new {a.mean():.1f} | " + " ".join(f"{x:.0f}" for x in a), flush=True); D_ = []
    for bn in env.BASES:
        base = A.summary(A.system(bn, A.FAMS)); s = A.summary(A.system(bn, fams, extra=[("c1x", 1.0)], slot="c1x"))
        dn, ds = s["dn"] - base["dn"], s["ds"] - base["ds"]; D_.append(dn); print(f"   {bn}: new {dn.mean():+.2f} seen {ds.mean():+.2f} (P04 {ds[0]:+.2f}, P10 {ds[1]:+.2f})", flush=True)
    m = np.mean(D_, 0); bt = np.percentile([m[rng.integers(0, 10, 10)].mean() for _ in range(4000)], [10, 90])
    print(f"   MEAN six bases: new {m.mean():+.2f} [{bt[0]:+.2f}, {bt[1]:+.2f}] people >= 0 {(m >= 0).sum()}/10 | " + " ".join(f"{x:+.1f}" for x in m), flush=True)
