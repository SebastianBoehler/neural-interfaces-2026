"""Marginal factor effects from a leaderboard json (random-design rows only, names R###): mean BA and ms per level,
with a config-resampling bootstrap 80 % interval of (level mean - factor mean). python effects.py LB.json"""
import json, sys
import numpy as np
lb = json.load(open(sys.argv[1]))
rows = [r for r in lb["rows"] if r["name"].startswith("R")]
ba = np.array([r["ba"] for r in rows]); ms = np.array([r["ms_per_window"] for r in rows])
rng = np.random.default_rng(0)
print(len(rows), "random-design configs; overall mean BA", round(ba.mean(), 2), "people", lb["people"], "orders", lb["orders"])
for f in ["steps", "lr", "every", "target", "aug", "lam_im", "params", "anchor"]:
    lv = np.array([str(r["cfg"][f]) for r in rows])
    out = []
    for l in sorted(set(lv), key=lambda s: float(s) if s.replace('.', '').replace('e-0', '').replace('-', '').isdigit() else s):
        m = lv == l; idx = np.flatnonzero(m)
        bs = [ba[rng.choice(idx, len(idx))].mean() - ba[rng.integers(0, len(ba), len(ba))].mean() for _ in range(2000)]
        out.append(f"{l}: {ba[m].mean()-ba.mean():+.2f} [{np.percentile(bs,10):+.1f},{np.percentile(bs,90):+.1f}] {ms[m].mean():.0f}ms")
    print(f"{f:8s}", " | ".join(out))
