"""Verify receipts and summarize 48 source-only Slurm runs."""

import argparse
import json
from pathlib import Path
from statistics import mean

from breadth_run import EVAL, METHODS, SEEDS

SCOPES = ("overall", "BrainHero", "Graz")


def read(path, method, seed, candidate, controls):
    run = json.loads(path.read_text())
    if (run["method"] != method or run["auxiliary_seed"] != seed or run["source_windows"] != 1323
            or set(run["protected_exclusions"]) != set(EVAL) or run["candidate"] != candidate
            or run["controls"] != list(controls) or set(run["scores"]) != {"0", "100", "200"}
            or set(run["audit"]) != {"a", "b", "c"}
            or any(set(fold["fit_people"]) & (set(EVAL) | set(fold["validation_people"]))
                   or set(fold["validation_people"]) & set(EVAL) for fold in run["audit"].values())):
        raise ValueError(f"Unexpected source receipt in {path}")
    return run["scores"]


def aggregate(folder, job):
    report = {}
    for method, (_, candidate, controls) in METHODS.items():
        seeds = {seed: read(folder / f"{method}-{seed}-{job}.json", method, seed, candidate, controls)
                 for seed in SEEDS}
        arms = seeds[SEEDS[0]]["0"].keys()
        by_seed = {seed: {arm: {k: mean(fold[arm][k]["delta"] for fold in scores.values())
                               for k in SCOPES} for arm in arms} for seed, scores in seeds.items()}
        averages = {arm: {k: mean(row[arm][k] for row in by_seed.values()) for k in SCOPES} for arm in arms}
        primary = averages[candidate]
        gains = {c: primary["overall"] - averages[c]["overall"] for c in controls}
        positive = sum(row[candidate]["overall"] > 0 for row in by_seed.values())
        checks = {"overall_0p8": primary["overall"] >= .8, "brainhero_0p5": primary["BrainHero"] >= .5,
                  "graz_nonnegative": primary["Graz"] >= 0, "matched_0p3": min(gains.values()) >= .3,
                  "five_positive_seeds": positive >= 5}
        report[method] = {"candidate": candidate, "means": averages, "by_seed": by_seed,
                          "matched_gains": gains, "pass": all(checks.values()), "checks": checks}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("array_job")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = aggregate(args.directory, args.array_job)
    temp = args.output.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2) + "\n")
    temp.replace(args.output)
