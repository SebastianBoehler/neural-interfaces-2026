"""Matched official cell accuracy and optional motor imagery recall."""
import numpy as np


PEOPLE = tuple(f"P{i:02d}" for i in range(1, 11))
ORDERS = ("within0", "within1", "within2", "within3", "full0", "full1", "runs0", "runs1")
COUNTS = np.array([2, 2, 2, 6, 2, 2, 2, 2, 2, 6])
STOCK_SCORE = 76.10391516713594


def motor_cells(meta, rows, labels, n_cells):
    cells = np.stack([meta["stream"][rows], meta["interface"][rows]], axis=1)
    _, index = np.unique(cells, axis=0, return_inverse=True)
    result = [tuple((index == i) & (labels == c) for c in range(3)) for i in range(n_cells)]
    if len(np.unique(index)) != n_cells or not all(m.any() for masks in result for m in masks):
        raise ValueError("motor recall requires every official cell/class")
    return result


def paired(interval, delta):
    return dict(**interval(delta, COUNTS),
                orders=dict(zip(ORDERS, np.average(delta, weights=COUNTS, axis=0).tolist())))


def summarize(tables, bank, pairs, interval, ref_kinds, eligible=None):
    result = {"absolute": {name: interval(table, COUNTS) for name, table in tables.items()}}
    contrasts = {}
    for left, right in pairs:
        changed = 0
        for p in PEOPLE:
            subset = slice(None) if eligible is None else eligible[p]
            for o in ORDERS:
                a, b = (bank[f"{p}|{o}|{arm}"][subset] for arm in (left, right))
                changed += np.count_nonzero(a != b)
        contrasts[f"{left}-{right}"] = dict(
            **paired(interval, tables[left] - tables[right]), changed=int(changed))
    for ref in ref_kinds:
        for scope in ("wave", "full"):
            genuine, blank = f"{scope}_genuine", f"{scope}_blank"
            delta = ((tables[genuine] - tables[blank])
                     - (tables[f"{ref}_{genuine}"] - tables[f"{ref}_{blank}"]))
            contrasts[f"{scope}_history_minus_{ref}_history"] = paired(interval, delta)
    result["contrasts"] = contrasts
    return result


def score_bank(bank, pairs, meta, interval, metric, labels, ref_kinds=(), evaluate_motor=False):
    names = sorted({name for pair in pairs for name in pair})
    tables = {name: np.empty((len(PEOPLE), len(ORDERS))) for name in names}
    motor_tables = {name: np.empty_like(table) for name, table in tables.items()} if evaluate_motor else {}
    eligible = {}
    for p, person in enumerate(PEOPLE):
        rows = meta["person"] == person
        y = labels[rows]
        cells = motor_cells(meta, rows, y, COUNTS[p]) if evaluate_motor else ()
        if evaluate_motor:
            eligible[person] = y == 0
        for i, order in enumerate(ORDERS):
            for name in names:
                pred = bank[f"{person}|{order}|{name}"]
                if pred.shape != y.shape or not np.isin(pred, (0, 1, 2)).all():
                    raise ValueError(f"invalid decision bank {person}/{order}/{name}")
                result = metric(y, pred, meta, rows)
                if result["n_cells"] != COUNTS[p]:
                    raise ValueError(f"unpaired metric cells {person}")
                tables[name][p, i] = 100 * result["balanced_accuracy_cell_mean"]
                if evaluate_motor:
                    recalls = np.asarray([[np.mean(pred[mask] == c) for c, mask in enumerate(cell)]
                                          for cell in cells])
                    if abs(recalls.mean() - result["balanced_accuracy_cell_mean"]) > 1e-12:
                        raise ValueError("class recall reduction differs from the official metric")
                    motor_tables[name][p, i] = 100 * recalls[:, 0].mean()
    result = summarize(tables, bank, pairs, interval, ref_kinds)
    if abs(result["absolute"]["stock"]["pp"] - STOCK_SCORE) > 1e-10:
        raise ValueError("stock metric mismatch")
    if evaluate_motor:
        result["motor_recall"] = summarize(motor_tables, bank, pairs, interval, ref_kinds, eligible)
    return result
