"""Collect the fresh-order quarter diet and check paired fold and delivery receipts."""
import argparse
import json
from pathlib import Path

import numpy as np

from oracles import freeze
from returning_diet_report import PARENT_SHA, digest, report
from quarter_protocol import CANDIDATE_SHA, CHUNK, ORDERS, ROSTER
from tapes import metadata, read_bank


def collect(data_path, fixtures, captures, output):
    files, receipts = {}, {}

    def read(fold, order, key):
        path = captures / f"{fold}-{order - order % CHUNK}.npz"
        if fold not in receipts:
            receipt_path = fixtures / "fixtures" / fold / "receipt.json"
            receipts[fold] = json.loads(receipt_path.read_text())[ROSTER]
        if path not in files:
            with np.load(path, allow_pickle=False) as tape:
                files[path] = dict(tape)
            if json.loads(str(files[path]["receipt"])) != receipts[fold]:
                raise ValueError(f"wrong fold receipt in {path}")
        tape = files[path]
        rows = tape[f"rows_{key}"]
        ref, trial = tape[f"half_{key}"], tape[f"quarter_{key}"]
        age, identity = (int(tape[f"{field}_{key}"]) for field in ("rec", "identity"))
        if any(labels.shape != rows.shape or labels.dtype != np.int8 or not np.isin(labels, (0, 1, 2)).all()
               for labels in (ref, trial)):
            raise ValueError(f"invalid label vectors in {path}/{key}")
        if not (age == -1 and identity == 0 or 1 <= age <= len(rows) and 1 <= identity <= 10):
            raise ValueError(f"invalid recognition receipt in {path}/{key}")
        prefix = age - 1 if age > 0 else len(rows)
        np.testing.assert_array_equal(trial[:prefix], ref[:prefix])
        return rows, dict(half=ref, quarter=trial, rec=np.full(len(rows), age), identity=np.full(len(rows), identity))

    bank = read_bank(metadata(data_path), ORDERS, read)
    output_doc = dict(orders=ORDERS, parent_sha256=PARENT_SHA, candidate_sha256=CANDIDATE_SHA,
                      captures_sha256={p.name: digest(p) for p in sorted(files)},
                      **report(bank, ORDERS, arms=("quarter",), nested_orders=True))
    freeze(output, output_doc)
    print(f"Collected {len(files)} shards into {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("data", "fixtures", "captures", "output"):
        parser.add_argument(arg, type=Path)
    a = parser.parse_args()
    collect(a.data, a.fixtures, a.captures, a.output)
