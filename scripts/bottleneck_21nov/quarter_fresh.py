"""Run the Fable R6 quarter diet against em_half on fresh shuffled orders."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from capture import top_gate
from returning_diet_worker import DATA_SHA, V, diet, fixture_assets, load, modified, open_fold, replay, taped
from returning_em_worker import sessions

from quarter_protocol import CHUNK, FOLDS, ORDERS, ROSTER, SCHEME


def run(data_path, fixtures, out, task, *, smoke=False):
    if not torch.cuda.is_available() or V.digest(data_path) != DATA_SHA:
        raise ValueError("CUDA and signed PROTEUS data required")
    if not 0 <= task < len(FOLDS) * len(ORDERS) // CHUNK:
        raise ValueError(f"task outside preregistered folds and orders: {task}")
    chunk, index = divmod(task, len(FOLDS))
    fold = FOLDS[index]
    data = V.D.load(data_path)
    signals = torch.from_numpy(data["X"])
    pkg, receipts = open_fold(fixtures, fold, "ba-confirm")
    receipt = receipts[ROSTER]
    people = fixture_assets(pkg / ROSTER, receipt)
    model, _ = load(data, pkg, ROSTER, receipt)
    streams = sessions(data, fold)
    start = chunk * CHUNK
    orders = ORDERS[start:start + CHUNK]
    payload = {"receipt": np.asarray(json.dumps(receipt, sort_keys=True))}
    if smoke:
        orders, streams = orders[:1], streams[:1]
    for order in orders:
        for stream, rows in V.D.plans(data, order, streams, scheme=SCHEME).items():
            items = taped(model, signals, rows)
            native = np.asarray([item.label for item in items], dtype=np.int8)
            np.testing.assert_array_equal(replay(model, items, "w25")[0], native)
            flags = np.asarray([item.recognised for item in items])
            age, identity = -1, 0
            half = quarter = native
            if flags.any():
                i = int(flags.argmax())
                gate = top_gate(items[i].contrast, items[i].identity)
                if not gate or gate[2] <= 0 or not flags[i:].all():
                    raise ValueError(f"invalid recognition latch: {fold}/{order}/{stream}")
                age, identity = i + 1, int(people[gate[0]][1:])
                half, half_scores, half_route = modified(model, items, "half", route=True)
                quarter, quarter_scores, quarter_route = diet(model, items, "quarter")
                np.testing.assert_array_equal(half[~flags], native[~flags])
                np.testing.assert_array_equal(quarter[~flags], half[~flags])
                np.testing.assert_array_equal(quarter_scores[~flags], half_scores[~flags])
                np.testing.assert_array_equal(quarter_route[~flags], half_route[~flags])
            key = f"{order}_{stream.split('|')[1]}"
            payload.update({f"rows_{key}": rows, f"half_{key}": half, f"quarter_{key}": quarter,
                            f"rec_{key}": np.asarray(age), f"identity_{key}": np.asarray(identity)})
            print(f"PASS {fold}/{key}: {len(rows)} rows, recognition {age}, identity {identity}", flush=True)
    if not smoke:
        with TemporaryDirectory(dir=out) as tmp:
            path = Path(tmp) / f"{fold}-{orders[0]}.npz"
            np.savez_compressed(path, **payload)
            path.rename(out / path.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "fixtures", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("task", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.data, args.fixtures, args.out, args.task, smoke=args.smoke)
