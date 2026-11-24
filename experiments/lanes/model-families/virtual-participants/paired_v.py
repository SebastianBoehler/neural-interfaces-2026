"""Lane V: paired full-system arm-minus-V0 (full0 laneC predictions of result_*-mNN.npz), people-resampled 80 %.
python laneV/paired_v.py DATA RESDIR"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "laneC"))
from oracle import metadata, metric_weights  # noqa: E402

d, res = metadata(Path(sys.argv[1])), Path(sys.argv[2]); y = d["y"]
people = np.unique(d["subject"]); W = {p: metric_weights(d, d["subject"] == p) for p in people}
rng = np.random.default_rng(20261119); samp = rng.integers(len(people), size=(10000, len(people)))
for m in ("m56", "m16"):
    base = np.load(res / f"result_V0-{m}.npz")["laneC"].argmax(1) if np.load(res / f"result_V0-{m}.npz")["laneC"].ndim > 1 else np.load(res / f"result_V0-{m}.npz")["laneC"]
    for a in ("V1", "V2w", "V3w"):
        z = np.load(res / f"result_{a}-{m}.npz"); p = z["laneC"]; p = p.argmax(1) if p.ndim > 1 else p
        by = np.array([100 * W[s][0] @ ((p == y).astype(int) - (base == y)) for s in people]); c = np.array([W[s][1] for s in people])
        ch = np.average(by[samp], weights=c[samp], axis=1)
        print(m, a, "minus V0", round(float(np.average(by, weights=c)), 3), np.quantile(ch, [.1, .9]).round(3).tolist())
