"""s257 reading: cost of capping = (capped arm + filter) minus (uncapped r15 + filter), from the npz that score.py (SFX=c,
ARMS=c100,c150,c250) saved: per order, per-person change vs ctl raw (= uncapped r15 raw). Scheme-equal mean, 80 % interval
over people and orders (as score.py), people up (person means over orders), per scheme.
Usage: python studies/14nov/stack/capcost.py runs/s242c_c100_c150_c250.npz"""
import sys
import numpy as np
z = np.load(sys.argv[1]); orders = sorted({k.split("|")[0] for k in z.files}); rng = np.random.default_rng(0)
by = {s: [o for o in orders if o.rstrip("0123456789") == s] for s in ("within", "full", "runs")}; by = {s: v for s, v in by.items() if v}
print(f"orders {len(orders)}: " + " ".join(f"{s} {len(v)}" for s, v in by.items()))
for n in ("c250", "c150", "c100"):
    D = {s: np.stack([z[f"{o}|{n}+filter"] - z[f"{o}|ctl+filter"] for o in v]) for s, v in by.items()}
    pt = np.mean([d.mean() for d in D.values()]); bt = []
    for _ in range(4000):
        pp = rng.integers(0, 10, 10); bt.append(np.mean([d[rng.integers(0, len(d), len(d))][:, pp].mean() for d in D.values()]))
    pm = np.mean([d.mean(0) for d in D.values()], 0); lo, hi = np.percentile(bt, [10, 90])
    raw = np.mean([np.stack([z[f"{o}|{n}"] for o in v]).mean() for s, v in by.items()])
    print(f"N{n[1:]}: cost with filter {pt:+.2f} [{lo:+.2f}, {hi:+.2f}] people up {int((pm > 0).sum())}/10 | " + " ".join(f"{s} {d.mean():+.2f}" for s, d in D.items()) + f" | raw (no filter) change vs uncapped {raw:+.2f}")
