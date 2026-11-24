"""Lane I returning collector (copy of Lane A's) (runs on the cluster, reads gold only here): balanced accuracy per (order, cell, arm), gains over
the stock control, and the intervals registered in evidence/wave_19nov/laneA_returning.md. Writes summary.json."""
import json
import sys
from pathlib import Path

import numpy as np

ORDERS = ("within0", "within1", "within2", "within3", "full0", "full1", "runs0", "runs1")
CELLS = ("P04|2", "P04|3", "P10|2", "P10|3")
ARMS = ("zero", "stock", "k20", "k10", "k5")
DATA = Path("/mnt/lustre/work/gehler/gfh527/ni26_aux431_20261118/data/proteus_windows.npz")


def bacc(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in range(3)]))


def main(out_dir, dest):
    with np.load(DATA, allow_pickle=False) as d:
        y = d["y"]
    ba, missing = {}, []
    for o in ORDERS:
        for c in CELLS:
            f = Path(out_dir) / f"{o}_{c.replace('|', '_')}" / f"{o}_{c.replace('|', '_')}.npz"
            if not f.exists():
                missing.append(f"{o}|{c}")
                continue
            with np.load(f) as z:
                for a in ARMS:
                    ba[(o, c, a)] = 100 * bacc(y[z["rows"]], z[f"{a}|labels"])
    done = [(o, c) for o in ORDERS for c in CELLS if (o, c, "stock") in ba]
    res = dict(missing=missing, n_cells=len(done), table={f"{o}|{c}|{a}": v for (o, c, a), v in ba.items()})
    if not missing:
        gain = {a: {(o, c): ba[(o, c, a)] - ba[(o, c, "stock")] for o, c in done} for a in ARMS[2:]}
        rng = np.random.default_rng(1190)
        res["k20_equals_stock_all"] = all(v == 0 for v in gain["k20"].values())
        res["zero_equals_stock_ba_diff"] = float(np.mean([ba[(o, c, "zero")] - ba[(o, c, "stock")] for o, c in done]))
        res["summary"] = {}
        for a in ARMS[2:]:
            g = np.array([[[gain[a][(o, f"{p}|{v}")] for v in (2, 3)] for o in ORDERS] for p in ("P04", "P10")])  # (2,8,2)
            per = g.mean(2)                                   # person x order
            eq = g.mean((0, 2)); w = 0.2 * per[0] + 0.8 * per[1]  # per order
            boots = []
            for _ in range(4000):
                ppl = rng.integers(0, 2, 2); ordr = rng.integers(0, 8, (2, 8))
                boots.append(np.mean([per[p][ordr[i]].mean() for i, p in enumerate(ppl)]))
            wb = [0.2 * per[0][rng.integers(0, 8, 8)].mean() + 0.8 * per[1][rng.integers(0, 8, 8)].mean() for _ in range(4000)]
            pb = [[per[p][rng.integers(0, 8, 8)].mean() for _ in range(4000)] for p in (0, 1)]
            res["summary"][a] = dict(
                equal_cell=float(g.mean()), w20_80=float(w.mean()), P04=float(per[0].mean()), P10=float(per[1].mean()),
                cells={c: float(np.mean([gain[a][(o, c)] for o in ORDERS])) for c in CELLS},
                orders_nonneg_w20_80=int((w >= 0).sum()),
                nested80_equal_cell=[float(np.percentile(boots, 10)), float(np.percentile(boots, 90))],
                order80_w20_80=[float(np.percentile(wb, 10)), float(np.percentile(wb, 90))],
                order80_P04=[float(np.percentile(pb[0], 10)), float(np.percentile(pb[0], 90))],
                order80_P10=[float(np.percentile(pb[1], 10)), float(np.percentile(pb[1], 90))])
    Path(dest).write_text(json.dumps(res, indent=1, sort_keys=True))
    print(json.dumps(res.get("summary", res), indent=1, sort_keys=True)[:3000])


if __name__ == "__main__":
    main(*sys.argv[1:])
