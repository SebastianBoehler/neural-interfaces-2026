"""gS vs stock collector (cluster, reads gold only here). Same statistics as laneA/collect.py, gain = gS minus stock.
Also: ctl minus stock must be exactly 0 on every cell, and gS must differ on some windows. Writes summary.json."""
import json
import sys
from pathlib import Path

import numpy as np

ORDERS = ("within0", "within1", "within2", "within3", "full0", "full1", "runs0", "runs1")
CELLS = ("P04|2", "P04|3", "P10|2", "P10|3")
DATA = Path("/mnt/lustre/work/gehler/gfh527/ni26_aux431_20261118/data/proteus_windows.npz")


def bacc(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in range(3)]))


def main(out_dir, dest):
    with np.load(DATA, allow_pickle=False) as d:
        y = d["y"]
    ba, changed, missing = {}, 0, []
    for o in ORDERS:
        for c in CELLS:
            t = f"{o}_{c.replace('|', '_')}"
            f = Path(out_dir) / t / f"{t}.npz"
            if not f.exists():
                missing.append(f"{o}|{c}")
                continue
            with np.load(f) as z:
                for a in ("stock", "ctl", "gS"):
                    ba[(o, c, a)] = 100 * bacc(y[z["rows"]], z[f"{a}|labels"])
                changed += int((z["gS|labels"] != z["stock|labels"]).sum())
    res = dict(missing=missing, windows_decision_changed=changed, table={f"{o}|{c}|{a}": v for (o, c, a), v in ba.items()})
    if not missing:
        res["ctl_minus_stock_max_abs"] = max(abs(ba[(o, c, "ctl")] - ba[(o, c, "stock")]) for o in ORDERS for c in CELLS)
        g = np.array([[[ba[(o, f"{p}|{v}", "gS")] - ba[(o, f"{p}|{v}", "stock")] for v in (2, 3)] for o in ORDERS] for p in ("P04", "P10")])
        per = g.mean(2)                                   # person x order
        w = 0.2 * per[0] + 0.8 * per[1]
        rng, B = np.random.default_rng(1911), 4000
        nested = [np.mean([per[p][rng.integers(0, 8, 8)].mean() for p in rng.integers(0, 2, 2)]) for _ in range(B)]  # people + orders
        wb = [0.2 * per[0][rng.integers(0, 8, 8)].mean() + 0.8 * per[1][rng.integers(0, 8, 8)].mean() for _ in range(B)]
        pb = [[per[p][rng.integers(0, 8, 8)].mean() for _ in range(B)] for p in (0, 1)]
        ci = lambda x: [float(np.percentile(x, 10)), float(np.percentile(x, 90))]
        res["summary"] = dict(
            equal_cell=float(g.mean()), w20_80=float(w.mean()), P04=float(per[0].mean()), P10=float(per[1].mean()),
            full_order_equal_cell=float(g[:, 4:6].mean()),
            cells={c: float(g[i // 2][:, i % 2].mean()) for i, c in enumerate(CELLS)},
            per_order_w20_80={o: float(w[i]) for i, o in enumerate(ORDERS)}, orders_nonneg_w20_80=int((w >= 0).sum()),
            nested80_equal_cell=ci(nested), order80_w20_80=ci(wb), order80_P04=ci(pb[0]), order80_P10=ci(pb[1]),
            full0_full1_w20_80=float(w[4:6].mean()))
    Path(dest).write_text(json.dumps(res, indent=1, sort_keys=True))
    print(json.dumps(res.get("summary", res), indent=1, sort_keys=True)[:3000], res.get("ctl_minus_stock_max_abs"), changed)


if __name__ == "__main__":
    main(*sys.argv[1:])
