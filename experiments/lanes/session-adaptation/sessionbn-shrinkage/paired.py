"""Lane I: paired full-system contrasts between replay prediction files (result_*.npz from laneH/full_replay.py), people-resampled 80 %.

python laneI/paired.py DATA.npz REPLAY_DIR OUT.json     contrasts: S1-S0, S2-S0, S0-stock, S1-stock, S2-stock per mass; plus per person and P04/P10."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "laneC")]
from oracle import metadata, metric_weights  # noqa: E402


def contrast(d, a, b, seed=20261119):
    y = d["y"]; people = {}
    for name in np.unique(d["subject"]):
        w, cells = metric_weights(d, d["subject"] == name)
        people[name.rsplit("/", 1)[-1]] = (float(100 * w @ ((b == y).astype(int) - (a == y))), cells)
    names = list(people); dl = np.array([people[n][0] for n in names]); ce = np.array([people[n][1] for n in names])
    s = np.random.default_rng(seed).integers(len(names), size=(10000, len(names)))
    ch = np.average(dl[s], weights=ce[s], axis=1)
    return dict(delta_pp=float(np.average(dl, weights=ce)), people80=np.quantile(ch, [0.1, 0.9]).tolist(), people80_lo_hi_p=float((ch <= 0).mean()),
                per_person={n: people[n][0] for n in names}, P04_P10_20_80=0.2 * people["P04"][0] + 0.8 * people["P10"][0])


def main():
    d = metadata(Path(sys.argv[1])); rd = Path(sys.argv[2]); out = {}
    for m in ("0.8", "0.1"):
        p = {}
        for arm in ("S0", "S1", "s2"):
            f = rd / f"result_{arm}-m{m}.npz"
            if f.exists():
                with np.load(f) as z:
                    p[arm] = z["laneC"].astype(int); p["stock"] = z["stock"].astype(int); p["zero"] = z["zero"].astype(int)
        assert (p["zero"] == p["stock"]).all(), "zero arm differs from stock"
        for key, base, new in (("S0-stock", "stock", "S0"), ("S1-stock", "stock", "S1"), ("s2-stock", "stock", "s2"),
                               ("S1-S0", "S0", "S1"), ("s2-S0", "S0", "s2")):
            if base in p and new in p:
                out[f"m{m}|{key}"] = contrast(d, p[base], p[new])
    Path(sys.argv[3]).write_text(json.dumps(out, indent=1) + "\n")
    for k, v in out.items():
        print(k, round(v["delta_pp"], 3), [round(x, 3) for x in v["people80"]], "P04", round(v["per_person"]["P04"], 2), "P10", round(v["per_person"]["P10"], 2))


if __name__ == "__main__":
    main()
