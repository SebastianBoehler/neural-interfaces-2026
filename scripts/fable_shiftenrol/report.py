"""Verify the registered returning comparisons and compute paired official scores."""
import argparse
import json
from pathlib import Path

import numpy as np

from fixtures import D, DATA_SHA, HELD, LEAD_SHA, PRECHECK, digest
from fit import SEED, half


def scores(data, people, orders, parts, phase):
    metrics, _ = D.official()
    by_fold = {}
    for path in parts:
        part = json.loads(path.read_text())
        fold = part["fold"]
        if fold in by_fold or part["data_sha"] != DATA_SHA or part["phase"] != phase or part["source"]["lead_sha"] != LEAD_SHA:
            raise ValueError(f"duplicate or unexpected experiment receipt: {path}")
        by_fold[fold] = part
    cells, activity = [], {}
    for person in people:
        part = by_fold.pop(f"ret_{person}")
        cases = {(c["visit"], c["order"]): c for c in part["cases"]}
        expected = {(v, o) for v in ("2", "3") for o in orders}
        if set(cases) != expected or len(part["cases"]) != len(expected):
            raise ValueError(f"incomplete experiment for {person}")
        activity[person] = {}
        for (visit, order), case in sorted(cases.items()):
            name = f"{person}|{visit}"
            rows = D.plans(data, order, [name])[name]
            if phase == "day":
                enrol, test = half(data, person, visit)
                source = {visit: enrol.tolist()}
                rows = rows[np.isin(rows, test)]
            else:
                source = {"1": np.flatnonzero(data["stream"] == f"{person}|1").tolist()}
            if case["source"] != source:
                raise ValueError(f"wrong enrolment rows for {name}")
            np.testing.assert_array_equal(rows, case["rows"])
            base, new = [np.array(case[k]) for k in ("base", "new")]
            for labels in (base, new):
                if labels.shape != rows.shape or not np.isin(labels, [0, 1, 2]).all():
                    raise ValueError(f"invalid decisions for {name}")
            activity[person][f"{visit}_{order}"] = {
                key: case[key] for key in ("base_meta", "new_meta", "weights_sha")
            }
            for interface in sorted(set(data["interface"][rows])):
                mask = data["interface"][rows] == interface
                y = data["y"][rows[mask]]
                cells.append(dict(person=person, visit=visit, order=order, interface=interface,
                                  n=int(mask.sum()), base=100 * metrics.balanced_accuracy(y, base[mask]),
                                  new=100 * metrics.balanced_accuracy(y, new[mask]),
                                  flips=int(np.count_nonzero(base[mask] != new[mask]))))
    if by_fold:
        raise ValueError("unexpected folds")
    return cells, activity


def summarize(cells, people, orders, phase):
    table = np.empty((len(people), len(orders), 2))
    for p, person in enumerate(people):
        for i, order in enumerate(orders):
            pairs = [(c["base"], c["new"]) for c in cells if (c["person"], c["order"]) == (person, order)]
            if len(pairs) != 4:
                raise ValueError(f"expected four cells for {person} order {order}")
            table[p, i] = np.mean(pairs, axis=0)
    delta = table[..., 1] - table[..., 0]
    rng = np.random.default_rng(SEED)
    draw_orders = rng.integers(len(orders), size=(20_000, len(orders)))
    draw_people = rng.integers(len(people), size=(20_000, len(people)))
    boot = delta[draw_people[:, :, None], draw_orders[:, None, :]].mean((1, 2))
    mean, individual = float(delta.mean()), delta.mean(1)
    return dict(base=float(table[..., 0].mean()), new=float(table[..., 1].mean()), delta=mean,
                ci80=np.quantile(boot, [.1, .9]).tolist(), per_order=delta.mean(0).tolist(),
                people={p: dict(base=float(table[i, :, 0].mean()), new=float(table[i, :, 1].mean()),
                                delta=float(individual[i]),
                                order_ci80=np.quantile(delta[i, draw_orders].mean(1), [.1, .9]).tolist())
                        for i, p in enumerate(people)},
                returning_20_80_unverified=float(individual @ [.2, .8]),
                advance=bool(mean >= 3 if phase == "day" else mean > 0 and min(individual) >= -.5))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("day", "held"))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("parts", type=Path, nargs=2)
    a = p.parse_args()
    if digest(a.data) != DATA_SHA:
        raise ValueError("unexpected EEG export")
    orders = PRECHECK if a.phase == "day" else HELD
    cells, activity = scores(D.load(a.data), D.RETURNING, orders, a.parts, a.phase)
    summary = summarize(cells, D.RETURNING, orders, a.phase)
    result = dict(phase=a.phase, orders=orders, sources={str(p): digest(p) for p in a.parts},
                  summary=summary, cells=cells, activity=activity)
    with a.out.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
