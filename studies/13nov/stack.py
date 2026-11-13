"""STACK (ledger s130): five replicated small levers for new people scored alone, all together and leave-one-out in the
routed chain (A.system), six bases, person bootstrap. Stored outputs only.
Usage: [BASES=B2_pool] [SMOKE=1] [SUBSETS=FT+OCU,...] python studies/13nov/stack.py"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stacklib import A, BASES, LEV, REPO, SMOKE, rng, run  # noqa: E402

ARMS = {"shipped": None, "base": ()}
ARMS.update({k: (k,) for k in LEV}); ARMS["ALL"] = LEV; ARMS.update({f"ALL-{k}": tuple(x for x in LEV if x != k) for k in LEV})
if os.environ.get("SUBSETS"):  # s130b: named subsets only, e.g. SUBSETS=REST+OCU,FT+REST+OCU
    SUB = {k: tuple(k.split("+")) for k in os.environ["SUBSETS"].split(",")}
    assert all(x in LEV for S in SUB.values() for x in S), f"unknown lever in {list(SUB)}: valid {LEV}"
    ARMS = {"shipped": None, "base": (), **SUB}
if SMOKE:
    ARMS = {k: ARMS[k] for k in ("shipped", "base", "FT", "PLS", "ALL")}; BASES = BASES[:1]
R = {}
for bn in BASES:
    t0 = time.time()
    for name, S in ARMS.items():
        R[bn, name] = run(S or (), bn, shipped=S is None)
    b = R[bn, "base"]
    print(f"{bn} ({time.time() - t0:.0f}s): base new {b[0].mean():.2f} seen {b[1].mean():.2f} | shipped new {R[bn, 'shipped'][0].mean() - b[0].mean():+.2f} seen {R[bn, 'shipped'][1].mean() - b[1].mean():+.2f} | "
          + " ".join(f"{k} {R[bn, k][0].mean() - b[0].mean():+.2f}/{R[bn, k][1].mean() - b[1].mean():+.2f}" for k in ARMS if k not in ("shipped", "base")), flush=True)


def summ(label, f):
    dn = np.array([f(bn, 0) for bn in BASES]); ds = np.array([f(bn, 1).mean() for bn in BASES]); d = dn.mean(0)
    bt = np.percentile([d[rng.integers(0, 10, 10)].mean() for _ in range(4000)], [10, 90])
    print(f"{label:14s} new {d.mean():+.2f} [{bt[0]:+.2f}, {bt[1]:+.2f}] bases +{(dn.mean(1) > 0).sum()}/{len(BASES)} ({dn.mean(1).min():+.2f} to {dn.mean(1).max():+.2f}) "
          f"people>=0 {(d >= 0).sum()}/10 P04 {d[3]:+.2f} | seen {ds.mean():+.2f}", flush=True)


print("\nagainst the stack base (ttt6 frozen SessionBN), new people, mean over bases")
for k in ARMS:
    if k != "base":
        summ(k, lambda bn, i, k=k: R[bn, k][i] - R[bn, "base"][i])
if not SMOKE and "ALL" in ARMS:
    print("\nleave-one-out cost (ALL minus ALL-without-lever)")
    for k in LEV:
        summ(f"keep {k}", lambda bn, i, k=k: R[bn, "ALL"][i] - R[bn, f"ALL-{k}"][i])
    print("\nALL against the shipped chain")
    summ("ALL-shipped", lambda bn, i: R[bn, "ALL"][i] - R[bn, "shipped"][i])
if not SMOKE and "ALL" not in ARMS:
    print("\nsubsets against the shipped chain")
    for k in SUB:
        summ(f"{k}-shipped", lambda bn, i, k=k: R[bn, k][i] - R[bn, "shipped"][i])
if not SMOKE:
    np.savez(REPO + f"/runs/stack_13nov{os.environ.get('STACK_TAG', '_sub' if os.environ.get('SUBSETS') else '')}.npz", **{f"{bn}|{k}|{j}": R[bn, k][j] for (bn, k) in R for j in (0, 1)})
