"""Collect and check all overnight source-fold comparisons."""

import json
from pathlib import Path
from statistics import mean
import sys

from night_run import EVAL, METHODS, SEEDS

SCOPES = ("overall", "BrainHero", "Graz")


def aggregate(folder):
    result = {}
    for method, (_, _, candidate, controls) in METHODS.items():
        reports = {}
        for seed in SEEDS:
            path = folder / f"{method}-{seed}.json"
            data = json.loads(path.read_text())
            if (data["method"] != method or data["auxiliary_seed"] != seed or data["source_windows"] != 1323
                    or tuple(data["protected_exclusions"]) != EVAL
                    or data["candidate"] != candidate or data["controls"] != list(controls)
                    or set(data["scores"]) != {"0", "100", "200"} or set(data["audit"]) != {"a", "b", "c"}
                    or any(set(fold["fit_people"]) & (set(EVAL) | set(fold["validation_people"]))
                           or set(fold["validation_people"]) & set(EVAL)
                           for fold in data["audit"].values())):
                raise ValueError(f"Unexpected source receipt in {path}")
            scores = data["scores"]
            reports[seed] = {arm: {k: mean(v[arm][k]["delta"] for v in scores.values()) for k in SCOPES}
                             for arm in scores["0"]}
        averages = {arm: {k: mean(v[arm][k] for v in reports.values()) for k in SCOPES}
                    for arm in reports[SEEDS[0]]}
        main = averages[candidate]
        gains = {c: main["overall"] - averages[c]["overall"] for c in controls}
        checks = {"overall_0p8": main["overall"] >= .8, "brainhero_0p5": main["BrainHero"] >= .5,
                  "graz_nonnegative": main["Graz"] >= 0, "control_0p3": min(gains.values()) >= .3,
                  "five_positive_seeds": sum(row[candidate]["overall"] > 0 for row in reports.values()) >= 5}
        result[method] = {"candidate": candidate, "means": averages, "seeds": reports,
                          "control_gains": gains, "pass": all(checks.values()), "checks": checks}
    return result


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    output = folder / "summary.json"
    temp = output.with_suffix(".tmp")
    temp.write_text(json.dumps(aggregate(folder), indent=2) + "\n")
    temp.replace(output)
