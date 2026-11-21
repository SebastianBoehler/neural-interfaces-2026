"""Freeze on fit orders and collect official paired 50/50 intervals."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from replay import DATA_SHA, DOSES, EARLY, ORDERS, PARENTS, V, sessions
from verify_collect import C, contrast

BINS = ((0, 40), (40, None))


def cell_weights(data, rows):
    interfaces, labels = data["interface"][rows], data["y"][rows]
    out = np.zeros(len(rows))
    for interface in np.unique(interfaces):
        cell = interfaces == interface
        groups = np.unique(labels[cell])
        for k in groups:
            mask = cell & (labels == k)
            out[mask] = 1 / (len(groups) * mask.sum())
    return out


def score_parts(run, data, roster, phase, arms):
    orders = ORDERS[phase]
    cells = {a: {} for a in arms}
    position = {a: {m: np.zeros((len(BINS), 2)) for m in ("new", "ret")} for a in arms[1:]}
    for mode, person in C.FOLDS:
        fold = f"{mode}_{person}"
        receipt = None
        for i in range(0, len(orders), 2):
            path = run / phase / roster / fold / f"{orders[i]}.npz"
            with np.load(path, allow_pickle=False) as tape:
                current = json.loads(str(tape["receipt"]))
                if (current["fold"] != fold or current["data_sha"] != DATA_SHA or
                        current["lead_sha"] != PARENTS["nine25"] or (receipt is not None and current != receipt) or
                        tuple(tape["arms"]) != arms or (current["partner"] is not None) != (roster == "ba_six")):
                    raise ValueError(f"unexpected roster, arms or receipt: {path}")
                receipt = current
                for order in orders[i:i + 2]:
                    for stream, rows in V.D.plans(data, order, sessions(data, fold)).items():
                        key = f"{order}_{stream.split('|')[1]}"
                        np.testing.assert_array_equal(tape[f"rows_{key}"], rows)
                        base = tape[f"w25_{key}"]
                        weights = cell_weights(data, rows)
                        y = data["y"][rows]
                        mask = np.zeros(len(data["y"]), dtype=bool)
                        mask[rows] = True
                        for arm in arms:
                            labels = tape[f"{arm}_{key}"]
                            if labels.shape != rows.shape or labels.dtype != np.int8 or not np.isin(labels, (0, 1, 2)).all():
                                raise ValueError(f"invalid labels in {path}: {arm}/{key}")
                            full = np.full(len(mask), -1)
                            full[rows] = labels
                            for c in V.D.cell_table(data, full, mask, order, mode):
                                cell = tuple(c[k] for k in C.KEYS)
                                if cell in cells[arm]:
                                    raise ValueError(f"duplicate cell {cell}")
                                cells[arm][cell] = c["ba"]
                            if arm != "w25":
                                delta = weights * ((labels == y).astype(int) - (base == y))
                                for b, (lo, hi) in enumerate(BINS):
                                    position[arm][mode][b] += (delta[lo:hi].sum(), weights[lo:hi].sum())
    if any(len(a) != 36 * len(orders) for a in cells.values()):
        raise ValueError("missing official cells")
    result = {a: contrast(cells[a], cells["w25"], orders) for a in arms}
    for a, entry in result.items():
        entry.pop("proxy_57_43_unverified")
        if a in position:
            entry["prefix_and_late_pp"] = {m: (100 * p[:, 0] / p[:, 1]).tolist() for m, p in position[a].items()}
    return result


def report(run, phase, data):
    best = json.loads((run / "fit.json").read_text())["rosters"] if phase == "held" else {}
    results = {}
    for roster in PARENTS:
        arms = (tuple(dict.fromkeys(("w25", best[roster]["weight"]))) if best else tuple(DOSES)) + tuple(EARLY)
        rows = score_parts(run, data, roster, phase, arms)
        weight = best[roster]["weight"] if best else max(tuple(DOSES), key=lambda a: rows[a]["j"])
        cold = best[roster]["cold"] if best else max(tuple(EARLY), key=lambda a: rows[a]["j"])
        def accepts(arm):
            return rows[arm]["j"] >= 0.15 and rows[arm]["j_ci80"][0] > 0

        results[roster] = dict(weight=weight, cold=cold, deltas=rows)
        if phase == "held":
            results[roster].update(weight_pass=accepts(weight), cold_accuracy_pass=accepts(cold))
        for a, row in rows.items():
            print(roster, a, f"{row['j']:+.3f} pp {row['j_ci80']}", flush=True)
    payload = dict(phase=phase, orders=ORDERS[phase], rosters=results)
    with TemporaryDirectory(dir=run) as scratch:
        path = Path(scratch) / "report.json"
        path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        path.rename(run / f"{phase}.json")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=tuple(ORDERS))
    p.add_argument("run", type=Path)
    p.add_argument("data", type=Path)
    args = p.parse_args()
    if V.digest(args.data) != DATA_SHA:
        raise ValueError("wrong PROTEUS export")
    report(args.run, args.phase, V.D.load(args.data))


if __name__ == "__main__":
    main()
