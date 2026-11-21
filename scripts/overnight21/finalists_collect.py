"""Finalists 21 Nov: seed robustness of Team B nine (orders 71124-71131) and the 16-order head-to-head (71200-71215).

python finalists_collect.py RUN_DIR --data NPZ --verify VERIFY_OUT --json OUT [--part seeds|h2h]
Each source is one arm directory of order chunks (<root>/<fold>/<arm>/pred_<mode>_<held>_<first order>.npz). Contrasts use the Lane TB
collector (people then orders resampled, official cell weights, 80 %). Missing chunks fail loudly. 57/43 is the UNVERIFIED sealed mix.
"""
import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "laneTB"))
import verify_collect as V  # noqa: E402

C, D = V.C, V.D
SEED_ORDERS = tuple(range(71124, 71132))
H2H_ORDERS = tuple(range(71200, 71216))


def stage(root, arm, name, orders, tmp):
    for mode, person in C.FOLDS:
        res = {}
        for f in sorted((root / f"{mode}_{person}" / arm).glob(f"pred_{mode}_{person}_*.npz")):
            with np.load(f, allow_pickle=False) as z:
                for k in z.files:
                    if k.startswith(("rows_", "labels_")):
                        assert k not in res, (f, k)
                        res[k] = z[k]
        got = {int(k.split("_")[1]) for k in res if k.startswith("rows_")}
        assert got == set(orders), (root, arm, mode, person, sorted(got))
        out = tmp / f"{mode}_{person}" / name
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / f"pred_{mode}_{person}.npz", source_sha="merged", weights_sha="merged", **res)


def cells_of(sources, orders, data):
    C.ORDERS = tuple(orders)
    cells, preds = {}, {}
    with TemporaryDirectory() as t:
        for name, (root, arm) in sources.items():
            stage(Path(root), arm, name, orders, Path(t))
            cells[name], preds[name], _ = C.load_arm(Path(t), name, data)
    return cells, preds


def mean_cells(items):
    return {k: float(np.mean([c[k] for c in items])) for k in items[0]}


def report(cells, pairs, orders, rows, result, tag):
    for a, b in pairs:
        key = f"{tag}: {a} − {b}"
        result[key] = V.contrast(cells[a], cells[b], orders)
        rows.append(V.line(f"{key} ({len(orders)} orders)", result[key]))


def changed(preds, a, b):
    n = sum(int(np.count_nonzero(preds[a][k] != preds[b][k])) for k in preds[a])
    return n, sum(len(v) for v in preds[a].values())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--data", required=True)
    p.add_argument("--verify", type=Path, required=True, help="laneTB-verify-21nov/out/hard (seed-0 tapes, orders 71124-71131)")
    p.add_argument("--json", type=Path, required=True)
    p.add_argument("--part", choices=("seeds", "h2h"), default="seeds")
    a = p.parse_args()
    data = D.load(a.data)
    rows, result = [], {}
    if a.part == "seeds":
        src = {"hard": (a.verify, "base")}
        for s, root in (("s0", a.verify), ("s1", a.run / "out/s1/hard"), ("s2", a.run / "out/s2/hard")):
            if not root.exists():
                print("missing", root)
                continue
            src.update({f"{s}_n25": (root, "nine25"), f"{s}_n15": (root, "nine15")})
        cells, _ = cells_of(src, SEED_ORDERS, data)
        seeds = [s for s in ("s0", "s1", "s2") if f"{s}_n25" in cells]
        for dose in ("n25", "n15"):
            cells[f"mean_{dose}"] = mean_cells([cells[f"{s}_{dose}"] for s in seeds])
            report(cells, [(f"{s}_{dose}", "hard") for s in seeds] + [(f"mean_{dose}", "hard")], SEED_ORDERS, rows, result, "seed")
            report(cells, [(f"{s}_{dose}", "s0_" + dose) for s in seeds[1:]], SEED_ORDERS, rows, result, "seed-pair")
        orders = SEED_ORDERS
    else:
        hh, g = a.run / "out/hh/hard", a.run / "finalists"
        src = {"hard": (hh, "base"), "n25": (hh, "nine25"), "n15": (hh, "nine15"),
               "g_hard": (g, "base"), "g_n25": (g, "nine25"), "ba": (g, "ba_terminal")}
        cells, preds = cells_of(src, H2H_ORDERS, data)
        for x, y in (("g_hard", "hard"), ("g_n25", "n25")):
            n, total = changed(preds, x, y)
            result[f"crosscheck {x} vs {y}"] = dict(changed=n, windows=total)
            rows.append(f"crosscheck: grow {x} vs verify {y}: {n}/{total} labels differ")
        pairs = [("n25", "hard"), ("n15", "hard"), ("ba", "hard"), ("ba", "n25"), ("n15", "n25"), ("ba", "n15")]
        report(cells, pairs, H2H_ORDERS, rows, result, "h2h")
        for half, orders in (("first8", H2H_ORDERS[:8]), ("last8", H2H_ORDERS[8:])):
            sub = {n: {k: v for k, v in c.items() if k[1] in orders} for n, c in cells.items()}
            report(sub, pairs[:4], orders, rows, result, half)
        orders = H2H_ORDERS
    a.json.write_text(json.dumps(dict(part=a.part, orders=list(orders), contrasts=result), indent=1) + "\n")
    print("| Contrast | Held-new Δ [80%] | Returning Δ [80%] | P04 / P10 | J | 57/43* | held-new per order |")
    print("|---|---|---|---|---|---|---|")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
