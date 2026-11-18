"""Export fit-excluded historical cap250 posteriors in source-row order."""

import json
from pathlib import Path

import numpy as np

from fit import DATA_SHA, ORDER_SHA, ORDERS, sha
from stocks456 import inputs, log_probability, REPORT_SHA, ref


def main():
    meta, ages, _, stock = inputs()
    if tuple(ref.ORDERS) != ORDERS:
        raise ValueError("unexpected §430 order names")
    age = np.stack([ages[order] for order in ORDERS]).astype(np.int64)
    scores = np.stack([log_probability(stock[order], ages[order]) for order in ORDERS])
    if scores.shape != (8, 2655, 3) or np.isnan(scores).any() or np.isposinf(scores).any():
        raise ValueError("invalid historical stock bank")
    if age.shape != (8, 2655) or np.count_nonzero(age == 0) != 8 * len(set(meta["stream"])):
        raise ValueError("invalid §430 stream age")
    scores = np.maximum(scores, np.log(1e-8)).astype(np.float32)
    if not np.array_equal(scores.argmax(-1), np.stack([stock[x].argmax(-1) for x in ORDERS])):
        raise ValueError("finite training floor changed stock labels")
    path = Path("teacher.npz")
    np.savez_compressed(path, score=scores, age=age, person=meta["person"])
    receipt = dict(bank=sha(path), data=DATA_SHA, orders_sha=ORDER_SHA,
                   source=REPORT_SHA, orders=ORDERS, windows=scores.shape[1])
    Path("teacher.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("§430 fit-excluded historical teacher rows, ages and decisions OK", flush=True)


if __name__ == "__main__":
    main()
