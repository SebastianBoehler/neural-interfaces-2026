"""Lane B6: full-system paired contrasts from laneH/full_replay.py result_<ARM>-m56/m16.npz (keys stock, zero, laneC, standalone).

python laneB6/paired.py DIR OUT.json   DIR holds result_{A,B,G,N,M,X}-m{56,16}.npz (and the .json for sanity). Per contrast: delta_pp,
people-only paired 80 % (full_replay.paired resampling, seed 20261119), P04 / P10 deltas alone, fixed / regressed windows.
Checks: every arm's stock equals A's stock (same cap250), zero == stock."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "laneC"))
from oracle import metadata, metric_weights  # noqa: E402

ARMS = "ABGNMX"
CONTRASTS = (("stock", "X"), ("stock", "M"), ("stock", "G"), ("stock", "N"), ("stock", "B"), ("stock", "A"), ("B", "X"), ("B", "M"), ("B", "G"), ("B", "N"), ("A", "B"), ("A", "X"), ("G", "M"), ("N", "M"), ("M", "X"))


def contrast(d, pa, pb):
    y = d["y"]; ok_a, ok_b = pa == y, pb == y
    w, _ = metric_weights(d, np.ones(len(y), bool))
    people = {n: metric_weights(d, d["subject"] == n) for n in np.unique(d["subject"])}
    delta = {n: (float(100 * wp @ (ok_b.astype(int) - ok_a)), cells) for n, (wp, cells) in people.items()}
    by = np.array([v[0] for v in delta.values()]); cells = np.array([v[1] for v in delta.values()])
    s = np.random.default_rng(20261119).integers(len(by), size=(10000, len(by)))
    iv = np.quantile(np.average(by[s], weights=cells[s], axis=1), [0.1, 0.9]).tolist()
    pick = lambda p: next(v[0] for n, v in delta.items() if n.endswith(p))
    return dict(delta_pp=float(w @ (ok_b.astype(int) - ok_a) * 100), people80=iv, P04=pick("P04"), P10=pick("P10"),
                fixed=int((~ok_a & ok_b).sum()), regressed=int((ok_a & ~ok_b).sum()))


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    d = metadata(Path(__file__).resolve().parents[1] / "data/proteus_windows.npz")
    rep = {}
    for m in ("m56", "m16"):
        pred = {}
        for a in ARMS:
            with np.load(root / f"result_{a}-{m}.npz") as z:
                pred[a] = z["laneC"]
                np.testing.assert_array_equal(z["zero"], z["stock"])
                if "stock" in pred: np.testing.assert_array_equal(pred["stock"], z["stock"])
                pred["stock"] = z["stock"]
        for a, b in CONTRASTS:
            rep[f"{b}-{a} {m}"] = contrast(d, pred[a], pred[b])
    out.write_text(json.dumps(rep, indent=1) + "\n")
    for k, v in rep.items():
        print(f"{k:16s} {v['delta_pp']:+.3f} {v['people80'][0]:+.3f} {v['people80'][1]:+.3f}  P04 {v['P04']:+.2f} P10 {v['P10']:+.2f}  +{v['fixed']}/-{v['regressed']}")


if __name__ == "__main__":
    main()
