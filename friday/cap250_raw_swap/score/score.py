"""Seal the F-C13 worker bank, then measure the registered Scherer contrasts."""

import argparse
import json
from pathlib import Path
import re

import numpy as np

from bank import ORDERS, locked_bank, metadata, seal_json, sha
from metrics import boot_draws, class_cells, contrast


BASE = Path("/mnt/lustre/work/gehler/gfh527")
DATA = BASE / "neural-interfaces-2026/data/scherer2015/windows.npz"
PRIOR = BASE / "ni26_cross411b_20261118"


def digest(value):
    if re.fullmatch("[0-9a-f]{64}", value) is None:
        raise argparse.ArgumentTypeError("expected a full lowercase SHA-256")
    return value


def collect(release, source, artifacts):
    work = release / "score"
    if (work / "result.json").exists():
        raise ValueError("the release already has a closed score")
    anchor_path = PRIOR / "result.json"
    archive = json.loads(anchor_path.read_text())
    subject, day, meta = metadata(DATA, archive)
    people = archive["people"]
    matrix, runtime, proof = locked_bank(release, PRIOR, source, artifacts, subject, day, meta["stream"])
    bindings = dict(scorer=sha(work / "score.sha256"), inputs=sha(work / "inputs.sha256"),
                    archive_result=sha(anchor_path))
    lock = seal_json(work / "before_labels.json", {**proof, **bindings})

    with np.load(DATA, allow_pickle=False) as data:
        labels = data["y"]
    classes, cells = class_cells(labels, matrix, subject, day, meta, people, archive)
    draws = boot_draws()
    arms = ("stock", *artifacts)
    contrasts = {arm: contrast(a, matrix, runtime, classes, cells, labels, people, draws)
                 for a, arm in enumerate(arms[1:], start=1)}
    primary = contrasts["seed0"]
    per_cell = {f"{person}|{visit}|{order}": dict(zip(arms, classes[:, i, v, j].tolist(), strict=True))
                for i, person in enumerate(people) for v, visit in enumerate((1, 2))
                for j, order in enumerate(ORDERS)}
    result = dict(
        metric="official equal-cell balanced accuracy", scope="FULL-SYSTEM EXTERNAL",
        cohort="reused §411b Scherer nine people, two visits, complete paired sessions",
        primary="seed0", replication="seed1" if "seed1" in contrasts else "pending",
        orders=ORDERS, rows_per_order=len(labels), bootstrap=dict(draws=10000, seed=468, level=.80),
        absolute_pct=dict(zip(arms, cells.mean(axis=(1, 2, 3)).tolist(), strict=True)),
        class_pct=dict(zip(arms, classes.mean(axis=(1, 2, 3)).tolist(), strict=True)),
        class_ids=(0, 1, 2), class_pct_by_cell=per_cell, contrasts=contrasts,
        primary_advance=primary["accuracy_outcome"] == "supports_gain" and primary["warm"]["passes_a100_gate"],
        primary_forecast=dict(pp=.20, minimum_changes_met=primary["decisions"]["changed"] >= 10,
                              net_rescue_positive=primary["decisions"]["net_rescued"] > 0),
        actor_calls=proof["actor_calls"], candidate_counter_changes=proof["candidate_counter_changes"],
        source=source, artifacts=artifacts, decision_lock=lock, **bindings,
    )
    result_sha = seal_json(work / "result.json", result)
    print(f"PASS F-C13 exact matched score {result_sha}", flush=True)
    print(json.dumps(dict(absolute_pct=result["absolute_pct"], contrasts=contrasts), sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("source", type=digest)
    parser.add_argument("seed0", type=digest)
    parser.add_argument("seed1", nargs="?", type=digest)
    opts = parser.parse_args()
    expected = {"seed0": opts.seed0}
    if opts.seed1:
        if opts.seed1 == opts.seed0:
            parser.error("seed 1 must have a distinct actor artifact")
        expected["seed1"] = opts.seed1
    collect(opts.release, opts.source, expected)
