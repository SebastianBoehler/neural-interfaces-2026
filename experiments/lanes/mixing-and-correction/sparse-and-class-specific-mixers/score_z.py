"""Official cell score and paired bootstrap of people and delivery orders."""
import json
import os
from pathlib import Path

import numpy as np

from bridge_owner import pin, sha, stamp, start_module
from complete_score import METRIC, METRIC_HASH
from complete_zero import FEAT, HASHES

WORK = Path(__file__).resolve().parent
ORDERS = ("full0", "within0")
ARMS = ("stock_base", "hardcb_base", "stock_portfolio", "hardcb_portfolio", "hardcb_additive", "hardcb_stack")
CONTRASTS = {
    "stock_portfolio_vs_stock": {"stock_portfolio": 1, "stock_base": -1},
    "hardcb_portfolio_vs_hardcb": {"hardcb_portfolio": 1, "hardcb_base": -1},
    "portfolio_raw_upgrade_interaction": {
        "hardcb_portfolio": 1, "stock_portfolio": -1, "hardcb_base": -1, "stock_base": 1,
    },
    "additive_vs_hardcb": {"hardcb_additive": 1, "hardcb_base": -1},
    "stack_vs_hardcb": {"hardcb_stack": 1, "hardcb_base": -1},
    "stack_vs_additive": {"hardcb_stack": 1, "hardcb_additive": -1},
}


def summarize(v, counts, people, draws, order_draws):
    draws_per_order = 100 * v[:, draws].sum(2) / counts[draws].sum(1)
    pooled_boot = draws_per_order[order_draws, np.arange(len(draws))].mean(0)
    return {
        "by_order": {o: float(s) for o, s in zip(ORDERS, 100 * v.sum(1) / counts.sum(), strict=True)},
        "by_person": {p: [float(x) for x in 100 * v[:, i] / counts[i]] for i, p in enumerate(people)},
        "pooled_pp": float(100 * v.mean(0).sum() / counts.sum()),
        "paired_80": np.percentile(pooled_boot, (10, 90)).tolist(),
        "separate_order_paired_80": np.percentile(draws_per_order, (10, 90), axis=1).T.tolist(),
    }


def main():
    pin(FEAT, HASHES[FEAT])
    pin(METRIC, METRIC_HASH)
    metric = start_module("lock_metric", METRIC, METRIC.read_text())
    with np.load(FEAT, allow_pickle=False) as z:
        y = z["y"].astype(int)
        subjects = z["subject"]
        groups = np.stack([subjects, z["session"], z["interface"]], axis=1)
    people = np.unique(subjects)
    masks = [subjects == p for p in people]
    counts = np.array([len(set(map(tuple, groups[m]))) for m in masks])
    assert len(people) == 10 and counts.sum() == 28
    rng = np.random.default_rng(260520)
    draws = rng.integers(0, len(people), (20000, len(people)))
    order_draws = rng.integers(0, len(ORDERS), (len(ORDERS), len(draws)))
    receipt, results = {}, {stage: {a: [] for a in ARMS} for stage in ("mix", "terminal")}
    for order in ORDERS:
        stem = WORK / f"{order}_complete"
        receipt[order] = row = json.loads(stem.with_suffix(".json").read_text())
        assert row["passed"] is True and row["rows"] == len(y) and row["source"] == sha(WORK / "source.sha256")
        # bridge_orders.write_npz returns the archive SHA-256.
        pin(stem.with_suffix(".npz"), row["output"]["sha256"])
        with np.load(stem.with_suffix(".npz"), allow_pickle=False) as z:
            assert stamp(z["delivery"]) == row["delivery"]
            for stage, arms in results.items():
                for arm in arms:
                    predicted = z[f"{arm}|{stage}"].argmax(1)
                    arms[arm].append([np.sum(metric.group_scores(metric.balanced_accuracy, y[m], predicted[m],
                                                                 groups[m])) for m in masks])
    report = {"receipts": receipt, "scorer_job": os.environ["SLURM_JOB_ID"], "scores": {}}
    for stage, arms in results.items():
        scores = {a: np.asarray(v) for a, v in arms.items()}
        scores.update({name: sum(k * scores[a] for a, k in weights.items())
                       for name, weights in CONTRASTS.items()})
        report["scores"][stage] = {name: summarize(v, counts, people, draws, order_draws)
                                    for name, v in scores.items()}
    path = WORK / "lock_scores.json"
    with path.open("x") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: [v["pooled_pp"], v["paired_80"]] for k, v in report["scores"]["terminal"].items()}))


if __name__ == "__main__":
    main()
