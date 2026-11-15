"""s271 step 0: leave-one-family-out marginals from the s269 files (alone reading, 6 seeds per family, all four vs the three left).
Marginal = BA(all 4 families, 24 nets) - BA(3 families, 18 nets); 80 % interval resamples people and (order, draw) cells.
Usage: python studies/14nov/ensemble_curve/lofo.py   (within0 within1 x d1 d2)"""
import itertools, os, sys
import numpy as np
sys.argv = sys.argv[:1]  # curve.py reads argv for orders; use its defaults
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import curve as C  # noqa: E402

N = C.load(); C.guard_single(N); cells = list(itertools.product(C.ORD, C.DRAWS)); nc = len(cells)
allnets = [(f, s) for f in C.FAM for s in C.SEEDS]; full = np.stack([C.subset_ba(N, o, d, allnets) for o, d in cells])
M = {}
for f in C.FAM:
    left = [(g, s) for g, s in allnets if g != f]; assert len(left) == 18 and all(g != f for g, _ in left)
    M[f] = full - np.stack([C.subset_ba(N, o, d, left) for o, d in cells])
M["mean of four"] = np.mean([M[f] for f in C.FAM], 0)
br = np.random.default_rng(1); idx = [(br.integers(0, len(C.PEOPLE), len(C.PEOPLE)), br.integers(0, nc, nc)) for _ in range(1000)]
L = [f"s271 step 0: leave-one-family-out marginals (all four, 24 nets, minus the three left, 18 nets); cells {cells}; {len(C.PEOPLE)} people", f"all four {full.mean():.2f} BA", ""]
for k, v in M.items():
    b = [v[np.ix_(oo, pp)].mean() for pp, oo in idx]; lo, hi = np.percentile(b, [10, 90])
    L.append(f"  marginal of {k:13s} {v.mean():+.2f}  [{lo:+.2f}, {hi:+.2f}]  people up {int((v.mean(0) > 0).sum())}/10  cells up {int((v.mean(1) > 0).sum())}/{nc}")
L.append("check: fbraw row equals s269 'drop fbraw' (+0.04 sign flipped: drop = left - all)")
txt = "\n".join(L); print(txt); os.makedirs(C.REPO + "/evidence/ensemble_curve_15nov", exist_ok=True); open(C.REPO + "/evidence/ensemble_curve_15nov/lofo.txt", "w").write(txt + "\n")
