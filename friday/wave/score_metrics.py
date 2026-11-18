"""Matched official cell accuracy and optional motor imagery recall."""
import numpy as np


PEOPLE = tuple(f"P{i:02d}" for i in range(1, 11))
ORDERS = ("within0", "within1", "within2", "within3", "full0", "full1", "runs0", "runs1")
COUNTS = np.array([2, 2, 2, 6, 2, 2, 2, 2, 2, 6])
STOCK_SCORE = 76.10391516713594
PREFIX_GROUPS = ("windows_1_10", "windows_11_40", "windows_41_plus")
SPLIT_PREFIX_GROUPS = ("window_1", "windows_2_10", *PREFIX_GROUPS[1:])


def class_cells(meta, rows, labels, n_cells):
    cells = np.stack([meta["stream"][rows], meta["interface"][rows]], axis=1)
    _, index = np.unique(cells, axis=0, return_inverse=True)
    result = [tuple((index == i) & (labels == c) for c in range(3)) for i in range(n_cells)]
    if len(np.unique(index)) != n_cells or not all(m.any() for masks in result for m in masks):
        raise ValueError("class diagnostics require every official cell/class")
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
            for o in ORDERS:
                subset = slice(None) if eligible is None else eligible[p, o]
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


def prefix_sets(ages, split_first=False):
    ages = np.asarray(ages)
    if ages.ndim != 1 or (ages < 0).any():
        raise ValueError("invalid delivery age for prefix diagnostics")
    late = ((ages >= 10) & (ages < 40), ages >= 40)
    return ((ages == 0), (ages > 0) & (ages < 10), *late) if split_first else (ages < 10, *late)


def cell_weights(cells, n_rows):
    weights = np.zeros(n_rows)
    for cell in cells:
        for mask in cell:
            weights[mask] = 1 / (3 * len(cells) * mask.sum())
    if not (weights > 0).all() or abs(weights.sum() - 1) > 1e-12:
        raise ValueError("paired prefix weights do not cover the official cell score")
    return weights


def score_bank(bank, pairs, meta, interval, metric, labels, ref_kinds=(), evaluate_motor=False,
               ages=None, split_first=False):
    names = sorted({name for pair in pairs for name in pair})
    tables = {name: np.empty((len(PEOPLE), len(ORDERS))) for name in names}
    motor_tables = {name: np.empty_like(table) for name, table in tables.items()} if evaluate_motor else {}
    groups = SPLIT_PREFIX_GROUPS if split_first else PREFIX_GROUPS
    prefix_tables = {group: {name: np.empty_like(table) for name, table in tables.items()}
                     for group in groups} if ages is not None else {}
    motor_eligible = {}
    prefix_eligible = {group: {} for group in prefix_tables}
    for p, person in enumerate(PEOPLE):
        rows = meta["person"] == person
        y = labels[rows]
        cells = class_cells(meta, rows, y, COUNTS[p]) if evaluate_motor or prefix_tables else ()
        weights = cell_weights(cells, len(y)) if prefix_tables else None
        for i, order in enumerate(ORDERS):
            if evaluate_motor:
                motor_eligible[person, order] = y == 0
            if prefix_tables:
                for group, subset in zip(groups, prefix_sets(ages[order][rows], split_first)):
                    prefix_eligible[group][person, order] = subset
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
                if prefix_tables:
                    correct = weights * (pred == y)
                    total = 0
                    for group, prefix in prefix_tables.items():
                        subset = prefix_eligible[group][person, order]
                        prefix[name][p, i] = 100 * correct[subset].sum()
                        total += prefix[name][p, i]
                    if abs(total - tables[name][p, i]) > 1e-10:
                        raise ValueError("prefix contributions do not sum to the official score")
    result = summarize(tables, bank, pairs, interval, ref_kinds)
    if abs(result["absolute"]["stock"]["pp"] - STOCK_SCORE) > 1e-10:
        raise ValueError("stock metric mismatch")
    if evaluate_motor:
        result["motor_recall"] = summarize(motor_tables, bank, pairs, interval, ref_kinds, motor_eligible)
    if prefix_tables:
        result["prefix_balanced_accuracy_contribution"] = {
            group: summarize(prefix, bank, pairs, interval, ref_kinds, prefix_eligible[group])
            for group, prefix in prefix_tables.items()
        }
    return result
