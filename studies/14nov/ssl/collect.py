"""s277 collect: person-equal BA per arm over the ten folds, arm minus arm 0, people up, P04 / P10. Usage: collect.py [runs/probe]"""
import glob, json, sys
import numpy as np
fs = sorted(glob.glob(f"{sys.argv[1] if len(sys.argv) > 1 else 'runs/probe'}/fold[0-9].json")); R = [json.load(open(f)) for f in fs]
assert len(R) == 10, f"{len(R)} folds"
ba = {a: np.array([r["arms"][str(a)]["ba"] for r in R]) * 100 for a in range(4)}; who = [r["held"].split("/")[-1] for r in R]
for r in R: assert r["steps"] == 6000 and r["nC"] == 26 and r["pool_subjects_md5"]["arm1"] != r["pool_subjects_md5"]["arm2"]
for a in range(4):
    dlt = ba[a] - ba[0]; p = lambda n: f"{dlt[who.index(n)]:+.1f}"
    print(f"arm{a}: BA {ba[a].mean():.2f}  minus arm0 {dlt.mean():+.2f}  people up {(dlt > 0).sum()}/10  P04 {p('P04')} P10 {p('P10')}")
print("per person (arm0..3):", {w: [round(float(ba[a][i]), 1) for a in range(4)] for i, w in enumerate(who)})
print("n_ext", {r["n_ext"] for r in R}, "n_pool1", sorted(r["n_pool1"] for r in R))
