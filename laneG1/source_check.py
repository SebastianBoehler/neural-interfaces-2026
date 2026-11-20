"""Source-only displacement-relatedness check (registered in evidence/wave_20nov/laneG1_emgraph.md). Local numpy on replay inputs.
usage (cd replay): python ../laneG1/source_check.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "replay"))
import paths  # noqa: F401
import numpy as np
import gm

Z = gm.L.Z
P = "all_loso"
F, y, sess, per = Z[f"GE|{P}"], Z[f"y|{P}"], Z[f"sess|{P}"], Z[f"person|{P}"]
iface = np.load(paths.WINDOWS)["interface"][Z[f"idx|{P}"]]
people = sorted(set(per))
z = np.empty_like(F)
for s in np.unique(sess):  # full-session standardization of the whole session (source analysis, no causality needed)
    r = sess == s; z[r] = (F[r] - F[r].mean(0)) / np.maximum(F[r].std(0), 1e-3)


def cm(rows):  # (3, d) class means, nan if a class is absent
    return np.stack([z[rows & (y == k)].mean(0) if (rows & (y == k)).any() else np.full(F.shape[1], np.nan) for k in range(3)])


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


EDGES = {"cog(1,2)": (1, 2), "wrongWM(2,0)": (2, 0), "wrongSM(1,0)": (1, 0)}
res = {e: {"cross": [], "within": []} for e in EDGES}
for p in people:
    A = np.nanmean([cm((per == q)) for q in people if q != p], 0)  # anchor: other people (equal person weight)
    ifs = sorted(set(iface[per == p]))
    D = {i: cm((per == p) & (iface == i)) - A for i in ifs}
    for e, (a, b) in EDGES.items():
        cr = [np.mean([cos(D[i][a], D[j][b]), cos(D[i][b], D[j][a])]) for i in ifs for j in ifs if i < j and np.isfinite(D[i]).all() and np.isfinite(D[j]).all()]
        wi = [np.mean([cos(D[i][a], D[i][b])]) for i in ifs if np.isfinite(D[i]).all()]
        res[e]["cross"].append(np.mean(cr) if cr else np.nan); res[e]["within"].append(np.mean(wi) if wi else np.nan)
rng = np.random.default_rng(7)
dr = rng.integers(0, len(people), (10000, len(people)))
for kind in ("cross", "within"):
    print(f"-- {kind}-interface relatedness (mean cos of class displacements; {len(people)} people; interfaces per person {[len(set(iface[per==p])) for p in people]})")
    v = {e: np.array(res[e][kind]) for e in EDGES}
    for e in EDGES:
        print(f"{e:14s} {np.nanmean(v[e]):+.3f}")
    for w in ("wrongWM(2,0)", "wrongSM(1,0)"):
        d = v["cog(1,2)"] - v[w]
        bs = np.nanmean(d[dr], 1)
        print(f"cog - {w}: {np.nanmean(d):+.3f}  people 80% [{np.percentile(bs, 10):+.3f}, {np.percentile(bs, 90):+.3f}]")
