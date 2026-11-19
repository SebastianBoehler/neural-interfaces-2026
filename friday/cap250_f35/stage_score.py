"""Read the raw, count-filtered and CSP scores from the signed F35c paths."""

import argparse
import json
from pathlib import Path

import numpy as np

from score_f35c import FEAT, HASHES, METRIC, METRIC_HASH, NAMES, ORDERS, WORK
from score_f35c import describe, held_bank, metrics, pin, sha, start_module, write_receipt


RESULT = "8fb06208095e6ea43598440884a67bb3ae9a71241a827111235dd2c9d5e20430"
BLIND = "76a70542f4891c44b7920e3e1adf8366bd4f175303e42d638db8dde3fe0607da"
STAGES = ("raw", "filtered", "terminal")
PARENTS = (("balance", "hard"), ("mirrored", "hard"), ("balance", "mirrored"))


def load_decisions(order, source, smoke, lock):
    p, proof, digest = held_bank(order, source, smoke)
    expected = lock["bank"][order]
    assert digest == expected["proof"] and proof["output"] == expected["output"]
    assert proof["folds"] == expected["inputs"]
    path = WORK / f"{order}.npz"
    pin(path, expected["output"]["sha256"])
    with np.load(path, allow_pickle=False) as bank:
        stages = {}
        for name in STAGES:
            scores = np.stack([bank[f"{arm}|{name}"] for arm in NAMES], axis=1)
            assert scores.shape == (2655, 4, 3) and np.isfinite(scores).all()
            np.testing.assert_array_equal(bank[f"stock|{name}"], bank[f"identity|{name}"])
            stages[name] = scores.argmax(2)
    np.testing.assert_array_equal(p, stages["terminal"])
    return stages, digest


def change_between(start, end, people, weights, sampled):
    result = {}
    for left, right in PARENTS:
        values = np.array([
            (end[p][left] - end[p][right]) - (start[p][left] - start[p][right]) for p in people
        ])
        simulated = np.average(values[sampled], weights=weights[sampled], axis=1)
        result[f"{left}_minus_{right}"] = dict(
            pp=float(np.average(values, weights=weights)),
            paired80=np.percentile(simulated, [10, 90]).tolist())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    pin(WORK / "results.json", RESULT)
    pin(WORK / "before-labels.json", BLIND)
    source = {key: sha(WORK / f"{key}.sha256") for key in ("pupil", "whole")}
    smoke = json.loads((WORK / "whole-smoke.json").read_text())
    lock = json.loads((WORK / "before-labels.json").read_text())
    orders = ("full0",) if args.smoke else ORDERS
    staged = {order: load_decisions(order, source, smoke, lock) for order in orders}
    output = args.output
    blind = output / ("smoke.json" if args.smoke else "before-labels.json")
    write_receipt(blind, source=sha(Path(__file__)), parent=BLIND,
                  proofs={order: proof for order, (_, proof) in staged.items()},
                  changes={order: {stage: (a[:, 2:] != a[:, [1]]).sum(0).tolist()
                                   for stage, a in decisions.items()}
                           for order, (decisions, _) in staged.items()},
                  held_labels_opened=False)
    if args.smoke:
        print("PASS F35d source and exact stage bank", sha(blind), flush=True)
        return
    pin(FEAT, HASHES[FEAT])
    pin(METRIC, METRIC_HASH)
    scorer = start_module("f35d_official", METRIC, METRIC.read_text())
    with np.load(FEAT, allow_pickle=False) as dataset:
        gold, subject, visit, interface = (dataset[k] for k in ("y", "subject", "session", "interface"))
    groups = np.stack((subject, visit, interface), axis=1)
    people = sorted(set(subject))
    weights = np.array([len(np.unique(groups[subject == person], axis=0)) for person in people])
    assert len(people) == 10 and weights.sum() == 28
    sampled = np.random.default_rng(460).integers(0, len(people), (4000, len(people)))
    original = json.loads((WORK / "results.json").read_text())["orders"]
    results = {}
    for order, (decisions, _) in staged.items():
        scores = {stage: describe(metrics(a, gold, subject, groups, scorer)) for stage, a in decisions.items()}
        assert scores["terminal"]["absolute_pct"] == original[order]["absolute_pct"]
        assert scores["terminal"]["contrasts"] == original[order]["contrasts"]
        cells = {stage: result["people_pct"] for stage, result in scores.items()}
        moves = {f"{end}_minus_{start}": change_between(cells[start], cells[end], people, weights, sampled)
                 for start, end in (("raw", "filtered"), ("filtered", "terminal"), ("raw", "terminal"))}
        results[order] = dict(stages=scores, moves=moves)
    score = output / "results.json"
    write_receipt(score, scope="Retrospective corrected F35c stage diagnostics", blind=sha(blind),
                  paired_draws=4000, seed=460, interval=.80, orders=results)
    print("PASS F35d", sha(score), flush=True)


if __name__ == "__main__":
    main()
