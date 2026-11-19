"""Validate exact order and completeness, then score the paired Scherer output arm."""

import json
from pathlib import Path
import sys

import numpy as np

from banks import digest, metadata, score
from orders import SEEDS, permutation

DATA_SHA = "a6188799753276451235678d08e92fd79162a95ec8099dbbb85e2ed9e2701e91"


def main(job, data):
    if digest(data) != DATA_SHA or (job / "replay.ok").read_text().strip() != "PASS":
        raise ValueError("missing prior qualification")
    subject, visit, cells = metadata(data)
    people = sorted(set(subject))
    n = len(subject)
    stock, trial = (np.full((len(SEEDS), n), -1, dtype=np.int8) for _ in range(2))
    source = {}
    for order, seed in enumerate(SEEDS):
        delivery = permutation(visit, None, "full", seed)
        for i, person in enumerate(people):
            path = job / "banks" / f"{len(SEEDS) * i + order}.npz"
            source[path.name] = digest(path)
            with np.load(path, allow_pickle=False) as bank:
                for key in np.unique(visit[subject == person]):
                    rows = bank[f"{key}|rows"]
                    np.testing.assert_array_equal(rows, delivery[visit[delivery] == key])
                    stock[order, rows], trial[order, rows] = bank[f"{key}|stock"], bank[f"{key}|labels"]
                    np.testing.assert_array_equal(bank[f"{key}|scores"].argmax(1), stock[order, rows])
    if not np.isin(stock, [0, 1, 2]).all() or not np.isin(trial, [0, 1, 2]).all():
        raise ValueError("incomplete bank")

    with (job / "blind_inputs.json").open("x") as fp:
        json.dump(source, fp, indent=2)
    with np.load(data, allow_pickle=False) as z:
        y = z["y"]
    result = score(y, stock, trial, subject, cells)
    result.update(n_people=len(people), seeds=SEEDS, before_gold=digest(job / "blind_inputs.json"))
    with (job / "result.json").open("x") as fp:
        json.dump(result, fp, indent=2)
        fp.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(*map(Path, sys.argv[1:]))
