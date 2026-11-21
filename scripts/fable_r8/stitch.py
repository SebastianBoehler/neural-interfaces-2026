"""Combine separately completed single-order repair tapes into the registered two-order shards."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from protocol import CHUNKS, FOLDS


FIELDS = {"rows", "hss", "nine25_fc", "ba_six_fp16", "em_half", "quarter", "terminal", "latch", "recognised"}
META = ("fold", "gpu", "direct")


def combine(repair, out, task):
    chunk, index = divmod(task, len(FOLDS))
    fold = FOLDS[index]
    orders = CHUNKS[chunk]
    paths = [repair / f"{fold}-{order}.npz" for order in orders]
    if not all(path.is_file() for path in paths):
        return False
    merged = {}
    for order, path in zip(orders, paths, strict=True):
        with np.load(path, allow_pickle=False) as tape:
            if str(tape["fold"]) != fold or not bool(tape["direct"]):
                raise ValueError(f"wrong live fold receipt in {path}")
            if merged and any(merged[key] != tape[key] for key in META):
                raise ValueError(f"device or fold mismatch in {path}")
            merged.update({key: tape[key] for key in META})
            suffixes = [field.removeprefix("rows_") for field in tape.files if field.startswith(f"rows_{order}_")]
            expected = {f"{field}_{suffix}" for field in FIELDS for suffix in suffixes}
            if not suffixes or set(tape.files) - set(META) != expected:
                raise ValueError(f"missing or non-matching order fields in {path}")
            merged.update({key: tape[key] for key in expected})
    name = paths[0].name
    with TemporaryDirectory(dir=out) as tmp:
        partial = Path(tmp) / name
        np.savez_compressed(partial, **merged)
        partial.rename(out / name)
    print(f"Stitched {name} from orders {orders}")
    return True


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("repair", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("tasks", nargs="+", type=int)
    a = p.parse_args()
    for task in a.tasks:
        if not combine(a.repair, a.out, task):
            raise SystemExit(f"both repair halves required for task {task}")
