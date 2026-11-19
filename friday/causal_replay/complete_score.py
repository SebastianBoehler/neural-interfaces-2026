"""Open held gold only after authenticating the two complete matched full0 systems."""
import argparse
import json
from pathlib import Path

import numpy as np

from bridge_owner import BASE, HERE, ROOT, pin, sha, start_module
from complete_zero import FEAT, FAMILIES, HASHES

KEYS = tuple(f"{arm}|{name}" for arm in ("historical", "corrected") for name in ("P0r", "cap250"))
METRIC = BASE / "neural-interfaces-2026/2026-competition/benchmark_utils/metrics.py"
METRIC_HASH = "fcc739400806a74259f248618501376a4fe53f2b5a40ccdc20db6a9e2369787d"


def authenticate(proof_sha):
    proof = HERE / "matched_full0.json"
    prediction = proof.with_suffix(".npz")
    pin(proof, proof_sha)
    receipt = json.loads(proof.read_text())
    if (receipt["order"] != "full0" or receipt["source_sha256"] != sha(HERE / "source.sha256")
            or not receipt["complete"] or receipt["held_labels_opened"]
            or set(receipt["matrix"]) != {f"{name}_s0" for name in FAMILIES}
            or not receipt["active_archive_prefix"]
            or any(v["exact_interleaved_prefix_checks"] != 3 or min(v["count_affected_prefix_rows"]) <= 0
                   for v in receipt["terminal"].values())):
        raise ValueError("full-system prediction seal, producer or complete-stage proof differs")
    pin(prediction, receipt["output_sha256"])
    with np.load(prediction, allow_pickle=False) as predicted, np.load(ROOT / "runs/shuffle_perms.npz") as order:
        if (not np.array_equal(predicted["index"], np.arange(2655))
                or not np.array_equal(predicted["delivery"], order["full|0|all_loso"])):
            raise ValueError("sealed rows or order differ")
        columns = np.column_stack([predicted[k] for k in KEYS])
        if not np.isin(columns, (0, 1, 2)).all():
            raise ValueError("invalid sealed class decisions")
        for i, key in enumerate(KEYS):
            suffix = "_terminal" if key.endswith("cap250") else "_posterior"
            if not np.array_equal(predicted[key + suffix].argmax(1), columns[:, i]):
                raise ValueError("sealed decisions differ from verified final stage")
    return columns


def score_columns(predicted, proof_sha, output, scope, order="full0"):
    pin(METRIC, METRIC_HASH)
    official = start_module("ni26_official_metrics", METRIC, METRIC.read_text())
    pin(FEAT, HASHES[FEAT])
    with np.load(FEAT, allow_pickle=False) as z:
        subject, visit, interface, labels = (z[k] for k in ("subject", "session", "interface", "y"))
    people = np.unique(subject)
    groups = np.stack((subject, visit, interface), axis=1)
    person_cells, sum_cells = [], []
    for person in people:
        selected = subject == person
        cells = np.column_stack([official.group_scores(official.balanced_accuracy,
                                 labels[selected], values[selected], groups[selected]) for values in predicted.T])
        person_cells.append(len(cells))
        sum_cells.append(cells.sum(0))
    sums, weights = np.array(sum_cells), np.array(person_cells)
    if len(people) != 10 or weights.sum() != 28:
        raise ValueError("unexpected held-new cell inventory")
    rng = np.random.default_rng(460)
    draw = rng.integers(0, len(people), (4000, len(people)))
    point = 100 * sums.sum(0) / weights.sum()
    bootstrap = 100 * sums[draw].sum(1) / weights[draw].sum(1)[:, None]
    contrasts = {"historical_cap250_minus_incumbent": np.array([-1, 1, 0, 0]),
                 "corrected_cap250_minus_incumbent": np.array([0, 0, -1, 1]),
                 "corrected_minus_historical_gain": np.array([1, -1, -1, 1])}
    result = dict(scope=scope, order=order,
                  absolute_balanced_accuracy_pct=dict(zip(KEYS, point)), proof_sha256=proof_sha,
                  people=list(people), official_cells=int(weights.sum()),
                  paired_person_bootstrap=dict(draws=len(draw), seed=460, interval=0.80),
                  contrasts={key: dict(pp=float(point @ v), paired80=np.percentile(bootstrap @ v, (10, 90)).tolist())
                             for key, v in contrasts.items()})
    with output.open("x") as handle:
        json.dump(result, handle, sort_keys=True, indent=2)
        handle.write("\n")
    output.chmod(0o444)



def score(proof_sha):
    score_columns(authenticate(proof_sha), proof_sha, HERE / "matched_full0_score.json",
                  "FULL-SYSTEM DEV, one held-new order, B2_pool")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("proof_sha")
    score(parser.parse_args().proof_sha)
