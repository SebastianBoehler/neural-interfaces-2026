"""Compare paired live terminal decisions using the official cell metric."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "scripts/laneTB")]
from verify_collect import C, contrast, line
from fixture import HELD, SCREEN, visits
from fit import DATA_SHA, D
from patch_terminal import LEAD_SHA, MIXES, verify_parent

BINS = ((0, 10), (10, 20), (20, 40), (40, None))


def cell_weights(data, rows):
    weights = np.zeros(len(rows), dtype=float)
    interface = data["interface"][rows]
    y = data["y"][rows]
    for cell in sorted(set(interface)):
        labels = set(y[interface == cell])
        for label in labels:
            sel = (interface == cell) & (y == label)
            weights[sel] = 1.0 / (len(labels) * sel.sum())
    return weights


def load(run, block, arms, orders, data):
    cells = {a: {} for a in arms}
    bands = {mode: {a: np.zeros((4, 2)) for a in arms[1:]} for mode in ("new", "ret")}
    receipts = {}
    for mode, person in C.FOLDS:
        fold = f"{mode}_{person}"
        for start in range(0, len(orders), 2):
            path = run / block / fold / f"{orders[start]}.npz"
            with np.load(path, allow_pickle=False) as tape:
                info = json.loads(str(tape["receipt"]))
                if (info["fold"] != fold or info["data_sha"] != DATA_SHA or info["lead_sha"] != LEAD_SHA or
                        info["arms"] != list(arms[1:]) or info != receipts.setdefault(fold, info)):
                    raise ValueError(f"wrong receipt for {path}")
                for o in orders[start:start + 2]:
                    for s, rows in D.plans(data, o, visits(data, fold)).items():
                        key = f"{o}_{s.split('|')[1]}"
                        np.testing.assert_array_equal(tape[f"rows_{key}"], rows)
                        lead = tape[f"lead_{key}"]
                        weights = cell_weights(data, rows)
                        y = data["y"][rows]
                        mask = np.zeros(len(data["y"]), dtype=bool)
                        mask[rows] = True
                        for a in arms:
                            got = tape[f"{a}_{key}"]
                            if got.shape != rows.shape or got.dtype != np.int8 or not np.isin(got, (0, 1, 2)).all():
                                raise ValueError(f"invalid labels for {path}/{a}/{key}")
                            full = np.full(len(mask), -1)
                            full[rows] = got
                            for c in D.cell_table(data, full, mask, o, mode):
                                cells[a][tuple(c[k] for k in C.KEYS)] = c["ba"]
                            if a != "lead":
                                gain = weights * ((got == y).astype(float) - (lead == y))
                                for i, (lo, hi) in enumerate(BINS):
                                    bands[mode][a][i] += (gain[lo:hi].sum(), weights[lo:hi].sum())
    for scores in cells.values():
        if len(scores) != 36 * len(orders):
            raise ValueError("wrong official cell count")
    bins = {m: {a: (100 * p[:, 0] / p[:, 1]).tolist() for a, p in by_arm.items()}
            for m, by_arm in bands.items()}
    return cells, bins, receipts


def admissible(summary):
    a, b = summary["new"]["mean"], summary["returning"]["mean"]
    return min(a, b) >= -0.30 and max(a, b) > 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("block", choices=("screen", "held"))
    p.add_argument("run", type=Path)
    p.add_argument("data", type=Path)
    a = p.parse_args()
    verify_parent(a.data, DATA_SHA)
    selected = None
    if a.block == "screen":
        arms, orders = ("lead", *MIXES), SCREEN
    else:
        selected = json.loads((a.run / "screen.json").read_text())["selected"]
        if selected is None:
            print("No candidate met the screen. Held-out orders remain unopened.")
            return
        arms, orders = ("lead", selected), HELD
    scores, bins, receipts = load(a.run, a.block, arms, orders, D.load(a.data))
    deltas = {name: contrast(scores[name], scores["lead"], orders) for name in arms[1:]}
    if a.block == "screen":
        possible = [name for name, value in deltas.items() if admissible(value)]
        selected = max(possible, key=lambda name: deltas[name]["j"], default=None)
    res = dict(orders=orders, selected=selected, deltas=deltas, delta_by_delivery_position=bins, fold_receipts=receipts)
    with (a.run / f"{a.block}.json").open("x") as f:
        json.dump(res, f, indent=1, allow_nan=False)
        f.write("\n")
    for name, value in deltas.items():
        print(line(name, value))
    print(f"Selected: {selected}", flush=True)


if __name__ == "__main__":
    main()
