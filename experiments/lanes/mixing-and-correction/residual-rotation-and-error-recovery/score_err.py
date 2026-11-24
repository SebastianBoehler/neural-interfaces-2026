"""Lane Y #3 gate: standalone BA (cell mean) W vs U, fixed-where-wrong rate ratio, people-resampled 80 %. python laneY/score_err.py err_full.npz"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
z = np.load(sys.argv[1]); y = z["y"]
d = np.load(ROOT / "data/proteus_windows.npz", allow_pickle=True); subj = d["subject"]; cells = np.array(["|".join(map(str, t)) for t in zip(subj, d["session"], d["interface"])])
people = sorted(set(subj)); wrong = {o: np.load(ROOT / f"work/laneZ/{o}_prof.npz")["stock|base|terminal"].argmax(1) != y for o in ("full0", "within0")}
def ba(g, p):  # per person: sum of cell BAs, count
    cs = [np.mean([(g[m & (y == c)] == c).mean() for c in range(3)]) for m in (((cells == k) & (subj == p)) for k in sorted(set(cells[subj == p])))]; return sum(cs), len(cs)
draw = np.random.default_rng(1).integers(0, 10, (10000, 10))
def stat(f):  # f(person) -> (num, den) ; returns point and 80 % interval of num/den pooled over people
    v = np.array([f(p) for p in people]); pt = v[:, 0].sum() / v[:, 1].sum(); b = v[draw][:, :, 0].sum(1) / v[draw][:, :, 1].sum(1); return pt, np.percentile(b, (10, 90))
R = {}
for a in ("W", "U"):
    g = z[a].argmax(1); R[a] = {"ba": stat(lambda p: ba(g, p))}
    for o in ("full0", "within0"):
        R[a][f"fww_{o}"] = stat(lambda p: ((g == y)[(subj == p) & wrong[o]].sum(), (subj == p)[wrong[o]].sum()))
        R[a][f"bwr_{o}"] = stat(lambda p: ((g != y)[(subj == p) & ~wrong[o]].sum(), ((subj == p) & ~wrong[o]).sum()))
for k in R["W"]:
    print(k, "W %.3f U %.3f ratio %.2f  W-U %.3f" % (R["W"][k][0], R["U"][k][0], R["W"][k][0] / R["U"][k][0], R["W"][k][0] - R["U"][k][0]))
# paired people bootstrap of the W-U BA difference
bw = np.array([ba(z["W"].argmax(1), p) for p in people]); bu = np.array([ba(z["U"].argmax(1), p) for p in people]); dd = bw[:, 0] - bu[:, 0]; n = bw[:, 1]
b = dd[draw].sum(1) / n[draw].sum(1); print("BA W-U pp %.2f [%.2f, %.2f]" % (100 * dd.sum() / n.sum(), *(100 * np.percentile(b, (10, 90)))))
