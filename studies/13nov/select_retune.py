"""Nested selection for RETUNE (ledger s166): choose the stage configuration on five people (mean new BA over the
shuffled schemes, bases and permutations), score it against the shipped constants on the other five, both ways.
Reports the shuffled gain and the recording-order change of the chosen configuration on the held-out half.
Usage: python studies/13nov/select_retune.py runs/night13/retune_*.npz"""
import sys
import numpy as np
R, grid = {}, None
for f in sys.argv[1:]:
    z = np.load(f); grid = z["grid"]; R.update({k: z[k] for k in z.files if k != "grid"})
cfgs = sorted({int(k.split("|")[0]) for k in R}); orders = sorted({tuple(k.split("|")[1:3]) for k in R}); bases = sorted({k.split("|")[3] for k in R})
assert len(cfgs) == len(grid), f"{len(cfgs)} of {len(grid)} configurations present: a job is missing"
SHIP = [i for i, g in enumerate(grid) if tuple(g) == (30, 1.15, 0.5, 60, 0.15, 60)][0]; sh = [o for o in orders if o[0] != "rec"]
print(f"{len(cfgs)} configurations, orders {orders}, bases {bases}; shipped = {SHIP}")


def score(c, people, ords):
    return np.mean([R[f"{c}|{o[0]}|{o[1]}|{b}"][people].mean() for o in ords for b in bases])


halves = (np.arange(0, 10, 2), np.arange(1, 10, 2))
for h, (sel, ev) in enumerate([(halves[0], halves[1]), (halves[1], halves[0])]):
    best = max(cfgs, key=lambda c: score(c, sel, sh)); g = grid[best]
    print(f"half {h}: chosen {best} (rebal W {g[0]:.0f} a {g[1]}, EM lam {g[2]} n0 {g[3]:.0f}, TSEM lam {g[4]} n0 {g[5]:.0f}) | held-out people: "
          f"shuffled {score(best, ev, sh) - score(SHIP, ev, sh):+.2f}, recording order {score(best, ev, [('rec', '0')]) - score(SHIP, ev, [('rec', '0')]):+.2f}", flush=True)
allp = np.arange(10)
top = sorted(cfgs, key=lambda c: -score(c, allp, sh))[:5]
print("top 5 on all ten (not nested; for the fresh-base check):")
for c in top:
    g = grid[c]
    print(f"  {c}: W {g[0]:.0f} a {g[1]} EM {g[2]}/{g[3]:.0f} TSEM {g[4]}/{g[5]:.0f} | shuffled {score(c, allp, sh) - score(SHIP, allp, sh):+.2f} rec {score(c, allp, [('rec', '0')]) - score(SHIP, allp, [('rec', '0')]):+.2f} | "
          + " ".join(f"{o[0]}{o[1]} {np.mean([R[f'{c}|{o[0]}|{o[1]}|{b}'].mean() - R[f'{SHIP}|{o[0]}|{o[1]}|{b}'].mean() for b in bases]):+.2f}" for o in orders))
