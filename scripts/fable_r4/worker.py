"""Run a no-score smoke or the true-label oracle on the registered fold deliveries."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from fold import DATA_SHA, D, FOLDS, LEAD, ORACLE, ORDERS, digest, pinned, stage, streams
from replay import replay


def oracle(data, rows):
    def prefix(age):
        return data["y"][rows[:age]].copy(), np.ones(age)
    return prefix


def smoke(data, template, fold, receipt):
    order = ORDERS[0]
    key = streams(data, fold)[0]
    rows = D.plans(data, order, [key])[key]
    expected, bank_sha, _ = pinned(fold, "nine25", order, key, rows)
    if bank_sha != receipt["bank_source_sha"]:
        raise ValueError("fold Team B bank does not match the pinned order")
    windows = data["X"][rows]
    zero = replay(template, windows)
    np.testing.assert_array_equal(zero["lead"], expected)
    truth = oracle(data, rows)
    tested = replay(template, windows, truth)
    np.testing.assert_array_equal(tested["lead"], expected)
    prefix = replay(template, windows[:110], truth)
    np.testing.assert_array_equal(prefix["enrolled"], tested["enrolled"][:110])
    np.testing.assert_array_equal(prefix["after100"], tested["after100"][:110])
    np.testing.assert_array_equal(prefix["ungated100"], tested["ungated100"][:110])
    np.testing.assert_allclose(prefix["loo"][:3], tested["loo"][:3], rtol=0, atol=0, equal_nan=True)
    print(json.dumps(dict(fold=fold, rows=len(rows), order=order, active=len(tested["active"]),
                          loo=np.nan_to_num(tested["loo"], nan=-1.).tolist(),
                          diagnostic_changes=int(np.count_nonzero(tested["ungated100"] != tested["lead"])),
                          recognized_at=int(tested["recognition"]), checkpoint_s=np.nan_to_num(tested["fit_seconds"]).tolist(),
                          **{k: v for k, v in receipt.items() if k != "fold"})), flush=True)
    if fold.startswith("new_"):
        if not np.isfinite(tested["loo"]).all() or not np.any(tested["ungated100"] != expected):
            raise ValueError("oracle did not fit at every checkpoint and exercise the terminal blend")
        if bool(len(tested["active"])) != bool((tested["loo"] >= .75).any()):
            raise ValueError("oracle acceptance disagrees with its leave-one-out guard")
    elif not (0 < tested["recognition"] <= 40) or len(tested["active"]):
        raise ValueError("returning smoke did not exercise the inactive recognized path")
    else:
        np.testing.assert_array_equal(tested["enrolled"], expected)
        np.testing.assert_array_equal(tested["ungated100"], expected)
    if template.team_b.windows or template.routeB.own.cur is not None:
        raise ValueError("template became dirty across copies")
    print("PASS exact pinned, disabled, active/inactive, prefix and session reset", flush=True)


def run(data, template, args, receipt):
    if not args.fold.startswith("new_"):
        raise ValueError("oracle requires a held-new fold")
    orders = ORACLE[2 * args.chunk:2 * args.chunk + 2]
    records = dict(orders=orders, **receipt)
    for order in orders:
        for key, rows in D.plans(data, order, streams(data, args.fold)).items():
            started = perf_counter()
            result = replay(template, data["X"][rows], oracle(data, rows))
            suffix = f"{order}_{key.split('|')[1]}"
            records[f"rows_{suffix}"] = rows
            records.update({f"{name}_{suffix}": value for name, value in result.items()})
            print(args.fold, suffix, len(rows), f"{perf_counter() - started:.1f}s",
                  "LOO", result["loo"], "fit s", result["fit_seconds"], flush=True)
    out = args.run / "oracle" / f"{args.fold}_{orders[0]}.npz"
    with out.open("xb") as f:
        np.savez_compressed(f, **records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=("smoke", "oracle"))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--fold", choices=FOLDS, required=True)
    p.add_argument("--lead", type=Path, default=LEAD)
    p.add_argument("--chunk", type=int, choices=(0, 1), default=0)
    args = p.parse_args()
    if digest(args.data) != DATA_SHA:
        raise ValueError("PROTEUS data SHA-256 differs")
    data = D.load(args.data)
    with TemporaryDirectory() as temp:
        template, receipt = stage(args.lead, data, args.fold, Path(temp))
        if args.task == "smoke":
            smoke(data, template, args.fold, receipt)
        else:
            run(data, template, args, receipt)


if __name__ == "__main__":
    main()
