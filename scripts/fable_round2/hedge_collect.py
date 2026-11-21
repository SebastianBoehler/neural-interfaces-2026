"""Official-cell score, preregistered arm freeze, and paired held-order intervals."""
import argparse
import json
from pathlib import Path

import numpy as np

from folds import DATA_SHA, D, FOLDS, streams
from hedge import CONFIRM, GRID, LEAD_ARM, SCHEDULES, SELECT
from live import digest
import collect as C


def load(run, phase, order_block, arms, data, selection):
    cells = {a: {} for a in arms}
    identities = {}
    for fold in FOLDS:
        mode, person = fold.split("_")
        for start in range(0, len(order_block), 2):
            orders = order_block[start:start + 2]
            with np.load(run / phase / f"{fold}_{orders[0]}.npz", allow_pickle=False) as tape:
                assert tuple(tape["orders"]) == orders and tuple(tape["arms"]) == tuple(arms)
                assert str(tape["data_sha"]) == DATA_SHA
                identity = {name: str(tape[name]) for name in ("source_sha", "weights_sha")}
                assert all(len(v) == 64 for v in identity.values())
                assert identities.setdefault(fold, identity) == identity
                if selection is not None:
                    assert str(tape["selection_sha"]) == selection
                for seed in orders:
                    mask = np.zeros(len(data["y"]), dtype=bool)
                    labels = {a: np.full(len(mask), -1, dtype=np.int8) for a in arms}
                    for stream, rows in D.plans(data, seed, streams(data, fold)).items():
                        key = f"{seed}_{stream.split('|')[1]}"
                        np.testing.assert_array_equal(rows, tape[f"rows_{key}"])
                        mask[rows] = True
                        for arm in arms:
                            pred = tape[f"{arm}_{key}"]
                            assert pred.shape == rows.shape and np.isin(pred, (0, 1, 2)).all()
                            labels[arm][rows] = pred
                    for arm in arms:
                        rows = D.cell_table(data, labels[arm], mask, seed, mode)
                        assert all(r["person"] == person for r in rows)
                        for r in rows:
                            key = tuple(r[k] for k in C.KEYS)
                            assert key not in cells[arm]
                            cells[arm][key] = r["ba"]
    for arm, c in cells.items():
        assert c.keys() == cells[LEAD_ARM].keys(), arm
        for order in order_block:
            assert sum(k[:2] == ("new", order) for k in c) == 28
            assert sum(k[:2] == ("ret", order) for k in c) == 8
    return cells, identities


def cohort(cells, mode, people, rng):
    values, counts = C.table(cells, mode, people)
    n, p = values.shape
    who = rng.integers(p, size=(20_000, p))
    order = rng.integers(n, size=(20_000, p, n))
    w = counts[who]
    draws = (values[order, who[:, :, None]].mean(2) * w).sum(1) / w.sum(1)
    per_order = np.average(values, axis=1, weights=counts)
    result = dict(mean=float(per_order.mean()), ci80=C.interval(draws),
                  by_person=dict(zip(people, values.mean(0).tolist(), strict=True)), per_order=per_order.tolist())
    return result, draws


def summarize(cells):
    rng = np.random.default_rng(71180)
    new, bn = cohort(cells, "new", D.PEOPLE, rng)
    ret, br = cohort(cells, "ret", D.RETURNING, rng)
    rp = ret["by_person"]
    return dict(new=new, returning=ret, j=(new["mean"] + ret["mean"]) / 2, j_ci80=C.interval((bn + br) / 2),
                proxy_57_43_unverified=0.57 * new["mean"] + 0.43 * (0.2 * rp["P04"] + 0.8 * rp["P10"]))


def sensitivity(contrasts):
    def dropped_score(cells, removed):
        return np.mean([np.mean([v for k, v in cells.items() if k[0] == cohort and k[2] != removed])
                        for cohort in ("new", "ret")])

    return {p: min(GRID, key=lambda a: (-dropped_score(contrasts[a], p), a)) for p in D.PEOPLE}


def record(run, phase, data):
    select_path = run / "select.json"
    previous = json.loads(select_path.read_text()) if phase == "confirm" else None
    orders = CONFIRM if previous else SELECT
    arms = [LEAD_ARM, previous["chosen"]] if previous else list(SCHEDULES)
    select_sha = digest(select_path) if previous else None
    cells, source = load(run, phase, orders, arms, data, select_sha)
    C.ORDERS = orders
    comparisons = {a: {k: v - cells[LEAD_ARM][k] for k, v in c.items()} for a, c in cells.items() if a != LEAD_ARM}
    summaries = {a: summarize(c) for a, c in comparisons.items()}
    result = dict(phase=phase, orders=list(orders), receipts=source, data_sha=DATA_SHA,
                  lead=summarize(cells[LEAD_ARM]), deltas=summaries)
    if previous:
        assert previous["receipts"] == source and previous["data_sha"] == DATA_SHA
        name = previous["chosen"]
        value = summaries[name]
        result.update(selection_sha=select_sha, chosen=name, config=GRID[name],
                      confirmed=all(value[k]["mean"] >= 0.30 and value[k]["ci80"][0] > 0
                                    for k in ("new", "returning")))
    else:
        name = min(GRID, key=lambda a: (-summaries[a]["j"], a))
        result.update(chosen=name, config=GRID[name], dropped_person_choice=sensitivity(comparisons))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("select", "confirm"))
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    args = p.parse_args()
    assert digest(args.data) == DATA_SHA
    result = record(args.run, args.phase, D.load(args.data))
    target = args.run / f"{args.phase}.json"
    with target.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print(args.phase, result["chosen"], digest(target), flush=True)


if __name__ == "__main__":
    main()
