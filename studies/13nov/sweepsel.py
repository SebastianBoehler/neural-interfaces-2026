"""SWEEPSEL (ledger s194): nested selection over the fine-tuning form sweep (ttt18 SWEEP=1, tag ...LS), RETUNE style.
For each configuration c, the combination (V6Fr) under it minus the combination under the in-run gC5 control, per person,
mean over bases and the given orders. A configuration is chosen on five people and scored on the other five (all 126
splits). Reported: per-configuration mean gain (all people and without P10), the held-out gain of the selection, and how
often each configuration is chosen.
Usage: python studies/13nov/sweepsel.py within0 within1 within2 within3"""
import glob, itertools, os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../.."); ORDERS = sys.argv[1:]
CF = ["gC5"] + [f"s{i:02d}" for i in range(1, 16)]
G = {}
for c in CF:
    rows = []
    for o in ORDERS:
        fs = sorted(glob.glob(f"{REPO}/runs/shufcombo_{o}_ocu_{c}_*ttt18{o}LS.npz")); f0 = sorted(glob.glob(f"{REPO}/runs/shufcombo_{o}_ocu_gC5_*ttt18{o}LS.npz"))
        assert len(fs) == 1 and len(f0) == 1, (o, c, fs, f0)
        a, b = np.load(fs[0]), np.load(f0[0]); bases = sorted({k.split("|")[0] for k in a.files})
        rows.append(np.mean([a[f"{bn}|V6Fr"] - b[f"{bn}|V6Fr"] for bn in bases], 0))
    G[c] = np.mean(rows, 0)
print(f"SWEEPSEL over {ORDERS}: combination under each form minus under gC5 (in-run control), new; mean / without P10")
for c in CF:
    print(f"  {c}: {G[c].mean():+.2f} / {np.delete(G[c], 9).mean():+.2f} | per person " + " ".join(f"{x:+.1f}" for x in G[c]))
gains, chosen = [], []
for half in itertools.combinations(range(10), 5):
    tr, te = list(half), [i for i in range(10) if i not in half]
    best = max(CF, key=lambda c: G[c][tr].mean()); chosen.append(best); gains.append(G[best][te].mean())
u, n = np.unique(chosen, return_counts=True)
print(f"NESTED: held-out gain of the selected form over gC5 {np.mean(gains):+.2f} [{np.percentile(gains, 10):+.2f}, {np.percentile(gains, 90):+.2f}] over 126 splits; chosen "
      + ", ".join(f"{x} {k}" for x, k in sorted(zip(u, n), key=lambda t: -t[1])))
