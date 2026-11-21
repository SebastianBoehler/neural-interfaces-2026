"""Collect the paired, row-authenticated arm c Scherer comparison."""
import argparse
from pathlib import Path

import numpy as np

from arm_c_sprint_policy import ARMS, CANDIDATE_SHA
from diet_decomposition_report import scherer
from oracles import freeze


def check_rows(data, manifest, captures, seeds):
    from niv import orders

    with np.load(data, allow_pickle=False) as source:
        subject, session = source["subject"], source["session"]
    runs = orders.runs_from_manifest(manifest, subject, session)
    people = sorted(set(subject))
    plans = {(scheme, i): orders.plans(subject, session, runs, scheme, seed)
             for scheme in ("full", "within") for i, seed in enumerate(seeds)}
    for arm in ("c", *ARMS):
        for n, person in enumerate(people):
            stream = f"{person}|2"
            path = captures / arm / f"ret_trial_k{n}" / f"p{n}_c{n % 4}.npz"
            with np.load(path, allow_pickle=False) as found:
                for (scheme, i), rows in plans.items():
                    np.testing.assert_array_equal(found[f"{scheme}|{i}|{stream}|rows"], rows[stream], err_msg=str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "manifest", "baseline", "captures", "out"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = scherer(args.data, args.baseline, args.captures, paired_a100=True, arms=ARMS,
                     reference="c", threshold=-0.10)
    if set(result["arms"]) != set(ARMS):
        raise ValueError("incomplete Scherer arm coverage")
    check_rows(args.data, args.manifest, args.captures, result["orders"])
    result["control_production_sha256"] = CANDIDATE_SHA
    freeze(args.out, result)


if __name__ == "__main__":
    main()
