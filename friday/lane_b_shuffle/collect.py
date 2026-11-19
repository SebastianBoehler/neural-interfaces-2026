"""Verify all paired decisions before opening gold and computing official cell scores."""

import json
from pathlib import Path
import sys

import numpy as np

from external import windows
from make import ARMS
from native import OFFICIAL
from orders import SCHEMES, SEEDS
from benchmark_utils.metrics import balanced_accuracy


NAMES = ("stock", *ARMS[1:])


def paired(cells, person, orders):
    rng = np.random.default_rng(19)
    delta = cells - cells[..., :1]
    people = np.unique(person)
    pd = rng.choice(people, (10000, len(people)))
    weights = (pd[..., None] == person).sum(axis=1)
    od = rng.integers(0, orders, (10000, orders))
    boot = (delta[od] * weights[:, None, :, None]).sum(axis=(1, 2)) / (orders * weights.sum(1)[:, None])
    result = {}
    for j, name in enumerate(NAMES):
        if j:
            result[name] = dict(stock_pct=float(cells[..., 0].mean()), score_pct=float(cells[..., j].mean()),
                                pp=float(delta[..., j].mean()), i80=np.percentile(boot[:, j], [10, 90]).tolist(),
                                i95=np.percentile(boot[:, j], [2.5, 97.5]).tolist(),
                                by_order=delta[..., j].mean(1).tolist(),
                                by_person={p: float(delta[:, person == p, j].mean()) for p in people})
    return result


def collect(data, manifest, root):
    _, people, visit, _, schedule = windows(data, manifest)
    unique = np.unique(people)
    cells = np.array(sorted(set(visit)))
    cell_people = np.array([v.split("|")[0] for v in cells])
    predictions = np.full((len(SCHEMES), len(SEEDS), len(people), len(NAMES)), -1, np.int8)
    expected = len(unique) * len(SCHEMES) * len(SEEDS)
    found = sorted(int(f.stem) for f in root.glob("[0-9]*.npz"))
    if found != list(range(expected)):
        raise ValueError(f"Incomplete external bank: {len(found)}/{expected}")
    meta = {name: [] for name in NAMES}
    for i in found:
        p, d = divmod(i, len(SCHEMES) * len(SEEDS))
        s, k = divmod(d, len(SEEDS))
        with np.load(root / f"{i}.npz", allow_pickle=False) as z:
            receipt = json.loads(z["receipt"].item())
            for v in cells[cell_people == unique[p]]:
                rows = z[f"{v}|rows"]
                plan = schedule[s, k]
                np.testing.assert_array_equal(rows, plan[visit[plan] == v])
                for j, name in enumerate(NAMES):
                    predictions[s, k, rows, j] = z[f"{v}|{name}"]
                    meta[name].append(dict(receipt["trials"][v][name], device=receipt["device"], scheme=SCHEMES[s]))
    if not np.isin(predictions, [0, 1, 2]).all():
        raise ValueError("External predictions are incomplete or invalid")
    np.testing.assert_array_equal(predictions[..., 0], predictions[..., NAMES.index("B4")])
    with np.load(data, allow_pickle=False) as z:
        y = z["y"]
    result = {}
    for s, scheme in enumerate(SCHEMES):
        ba = np.empty((len(SEEDS), len(cells), len(NAMES)))
        for j, v in enumerate(cells):
            group = visit == v
            for k in range(len(SEEDS)):
                ba[k, j] = [100 * balanced_accuracy(y[group], p) for p in predictions[s, k, group].T]
        result[scheme] = paired(ba, cell_people, len(SEEDS))
        for j, name in enumerate(NAMES[1:], 1):
            result[scheme][name]["changed"] = int(np.count_nonzero(predictions[s, ..., j] != predictions[s, ..., 0]))
    return dict(n_people=len(unique), n_sessions=len(cells), orders_per_scheme=len(SEEDS), result=result, work=meta)


if __name__ == "__main__":
    data, runs, root, output = map(Path, sys.argv[1:])
    with output.open("x") as fp:
        json.dump(collect(data, runs, root), fp, allow_nan=False, indent=2)
        fp.write("\n")
