"""Terminal pseudo-label quality, source-only selectors and official-cell class counts."""

import csv
import json
from pathlib import Path

import numpy as np

from balance import terminal_probability
from banks import DATA, ORDERS, ROOT, digest, metadata, stock_bank, verify_manifest

MARGINS = np.arange(0, 100, 5) / 100


def members(order, delivery, paths):
    votes = []
    for path in paths:
        with np.load(path, allow_pickle=False) as bank:
            if "orders" in bank:
                i = bank["orders"].tolist().index(order)
                plan, probabilities = bank["delivery"][i], bank["proba"][i]
            elif path.stem == f"fbl_{order}":
                plan, probabilities = bank["delivery"], bank["proba"]
            else:
                continue
            np.testing.assert_array_equal(bank["index"], np.arange(len(delivery)))
            np.testing.assert_array_equal(plan, delivery)
            if not np.isfinite(probabilities).all():
                raise ValueError(f"invalid family bank: {path}")
            votes.append(probabilities.argmax(1))
    if len(votes) != 4:
        raise ValueError(f"expected four delivery-matched member families for {order}")
    return np.stack(votes, axis=1)


def stats(mask, correct):
    n = int(mask.sum())
    return dict(n=n, coverage=n / len(mask), purity=float(correct[mask].mean()) if n else None)


def rule(source, m, agree, correct):
    choices = []
    for k in range(1, 5):
        for margin in MARGINS:
            mask = (agree >= k) & (m > margin) & source
            n = int(mask.sum())
            if n >= 30 and correct[mask].mean() >= .88:
                choices.append((n, k, float(margin)))
    return max(choices) if choices else None


def counts(y):
    return np.bincount(y, minlength=3).tolist()


def quota(rows, selection, pred, margin):
    groups = [rows[selection[rows] & (pred[rows] == c)] for c in range(3)]
    n = min(map(len, groups))
    return np.concatenate([r[np.argsort(-margin[r], kind="stable")[:n]] for r in groups])


def measure(order, subject, visit, cell, y, paths):
    delivery, pred, terminal, raw = stock_bank(order, len(y))
    p = terminal_probability(terminal)
    conf = p.max(1)
    confidence = np.sort(p, axis=1)
    margin = confidence[:, -1] - confidence[:, -2]
    agree = (members(order, delivery, paths) == pred[:, None]).sum(1)
    correct = pred == y
    fraction = np.empty(len(y))
    selection = np.zeros(len(y), dtype=bool)
    quota_selection = selection.copy()
    masks = np.ones(len(y), dtype=bool)
    envelope = rule(masks, margin, agree, correct)
    best = (agree >= envelope[1]) & (margin > envelope[2]) if envelope else ~masks
    rules, quotas = {}, {}
    for person in np.unique(subject):
        source = subject != person
        chosen = rule(source, margin, agree, correct)
        rules[str(person)] = chosen
        if chosen:
            selection |= ~source & (agree >= chosen[1]) & (margin > chosen[2])
    for key in np.unique(visit):
        rows = delivery[visit[delivery] == key]
        fraction[rows] = np.arange(len(rows)) / len(rows)
        subset = quota(rows, selection, pred, margin)
        quota_selection[subset] = True
        quotas[str(key)] = dict(all=len(rows), quota_selected=len(subset),
                                quota_selected_by_age190=len(quota(rows[:190], selection, pred, margin)),
                                age190_reached=bool(len(rows) >= 190))

    confidence_grid = [0.0, .45, .55, .65, .75, .85, 1.01]
    ages = np.minimum((fraction * 5).astype(int), 4)
    benchmark = {
        "terminal": stats(masks, correct),
        "stock_upstream_raw": stats(masks, raw.argmax(1) == y),
        "confidence": {f"[{a:.2f},{b:.2f})": stats((a <= conf) & (conf < b), correct)
                       for a, b in zip(confidence_grid, confidence_grid[1:])},
        "agreement_exact": {k: stats(agree == k, correct) for k in range(5)},
        "fifths": {i + 1: stats(ages == i, correct) for i in range(5)},
        "fifth_by_agreement_margin": {
            i + 1: {f"{k}:{m:.2f}": stats((ages == i) & (agree >= k) & (margin > m), correct)
                    for k in range(5) for m in (0.0, .20, .40, .60, .80)} for i in range(5)
        },
        "held_descriptive_envelope": dict(rule=envelope, **stats(best, correct)),
        "other_people_threshold": dict(rules=rules, **stats(selection, correct)),
        "other_people_threshold_equal_quotas": dict(visits=quotas, **stats(quota_selection, correct))
    }
    detail = []
    for key in np.unique(cell):
        here = cell == key
        for name, mask in (("stock", masks), ("held_envelope", best),
                           ("source_selected", selection), ("source_selected_quota", quota_selection)):
            selected = here & mask
            detail.append([order, key, name, int(here.sum()), int(selected.sum()), int(correct[selected].sum()),
                           *counts(pred[selected]), *counts(y[selected])])
    return benchmark, detail


def main(output):
    verify_manifest()
    subject, visit, cell = metadata()
    pathset = []
    for line in Path(__file__).with_name("member_inputs.sha256").read_text().splitlines():
        expected, path = line.split()
        path = ROOT / path
        if digest(path) != expected:
            raise ValueError(f"member bank changed: {path}")
        pathset.append(path)
    with np.load(DATA, allow_pickle=False) as z:
        y = z["y"]
    result, rows = {}, []
    for order in ORDERS:
        result[order], detail = measure(order, subject, visit, cell, y, pathset)
        rows.extend(detail)
    with (output / "purity.json").open("x") as fp:
        json.dump(result, fp, indent=2)
        fp.write("\n")
    with (output / "purity_cells.csv").open("x") as fp:
        writer = csv.writer(fp, lineterminator="\n")
        writer.writerow(["order", "cell", "selection", "n_cell", "n_selected", "correct",
                         "pseudo_0", "pseudo_1", "pseudo_2", "true_0", "true_1", "true_2"])
        writer.writerows(rows)


if __name__ == "__main__":
    import sys

    main(Path(sys.argv[1]))
