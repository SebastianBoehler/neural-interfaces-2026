"""Score probability-averaging ensembles from saved per-window probabilities.

Usage: python scripts/combine_probs.py OUT.json PROTOCOL [PROBS_DIR ...]
Reads NAME-PROTOCOL-sSEED.npz files written by experiments/run_cv.py and
scores, with the official metric, the same weighted mean that
experiments/ensemble.py computes:
  per seed   v2 + one deep member (seed s), mean and SD over seeds;
  seed-avg   v2 + the mean of all seeds of a deep member ("x5");
  weights    v2 weight 0.3-0.7 against the seed-averaged member.
Selection uses dev_loso/dev_session only; all_loso is evaluation only.
"""

import json
import sys
from pathlib import Path
from statistics import mean, stdev

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import protocol  # noqa: E402

DEEP = ("shallow_ea", "shallow_raw")


def load(dirs, name, proto):
    out = {}
    for d in dirs:
        for f in sorted(Path(d).glob(f"{name}-{proto}-s*.npz")):
            with np.load(f) as z:
                out[int(f.stem.rsplit("-s", 1)[1])] = (z["proba"], z["index"])
    return out


def main():
    out_path, proto, dirs = sys.argv[1], sys.argv[2], sys.argv[3:]
    dirs = dirs or [ROOT / "runs/experiments/probs"]
    d = protocol.load()
    v2, idx = load(dirs, "v2", proto)[0]
    mask = np.zeros(len(d["y"]), bool)
    mask[idx] = True
    y = d["y"][mask]

    def cell(P):
        return protocol.score(y, P.argmax(1), d, mask)["balanced_accuracy_cell_mean"]

    res = {"protocol": proto, "v2": cell(v2)}
    deep = {}
    for name in DEEP:
        runs = load(dirs, name, proto)
        assert all((i == idx).all() for _, i in runs.values()), name
        deep[name] = {s: p for s, (p, _) in sorted(runs.items())}
        res[f"seeds_{name}"] = sorted(deep[name])
        res[name] = [cell(p) for p in deep[name].values()]
        res[f"v2+{name}"] = [cell((v2 + p) / 2) for p in deep[name].values()]
        avg = np.mean(list(deep[name].values()), 0)
        res[f"{name}_x5"] = cell(avg)
        res[f"v2+{name}_x5"] = cell((v2 + avg) / 2)
        res[f"v2+{name}_x5_weights"] = {
            str(w): cell(w * v2 + (1 - w) * avg) for w in (0.3, 0.4, 0.5, 0.6, 0.7)}
    common = sorted(set(deep[DEEP[0]]) & set(deep[DEEP[1]]))
    res["v2+both"] = [cell((v2 + deep[DEEP[0]][s] + deep[DEEP[1]][s]) / 3)
                      for s in common]
    avg2 = [np.mean(list(deep[n].values()), 0) for n in DEEP]
    res["v2+both_x5"] = cell((v2 + avg2[0] + avg2[1]) / 3)
    Path(out_path).write_text(json.dumps(res, indent=1) + "\n")
    for k, v in res.items():
        if isinstance(v, list) and v and isinstance(v[0], float):
            sd = stdev(v) if len(v) > 1 else 0.0
            print(f"{k:26s} {mean(v):.4f} ± {sd:.4f} (n={len(v)})")
        elif isinstance(v, float):
            print(f"{k:26s} {v:.4f}")
        elif isinstance(v, dict):
            print(f"{k:26s} " + " ".join(f"{w}:{s:.4f}" for w, s in v.items()))


if __name__ == "__main__":
    main()
