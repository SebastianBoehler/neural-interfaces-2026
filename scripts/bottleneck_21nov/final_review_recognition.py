"""Check Fable's early-only oracle on the registered fit block."""

import argparse
import json
from pathlib import Path

import numpy as np

from final_review_score import ORDERS, ROSTER
from returning_em_report import load_cases
from tapes import metadata, paired, score, stages
from terminal import check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("work", type=Path)
    args = parser.parse_args()
    info = metadata(args.data)
    orders = ORDERS["fit"]
    cases = stages(info, args.work, ROSTER, orders)
    check(cases, ROSTER)
    legacy = {c.key: c for c in load_cases(args.work, info, ROSTER, "fit")}
    for case in cases:
        rec = legacy[case.key].scores["rec"]
        flags = (rec > 0) & (np.arange(len(rec)) >= rec - 1)
        np.testing.assert_array_equal(flags, case.scores["recognised"])
        if case.cohort == "ret" and not flags.any():
            raise ValueError("the registered returning instrument has no latch")

    def reference(case):
        return legacy[case.key].scores["half"]

    result = {}
    for first in (1, 3, 5, 8, 12):
        def candidate(case):
            native = reference(case)
            if case.cohort != "ret":
                return native
            age = np.arange(1, len(native) + 1)
            use = (first <= age) & (age < legacy[case.key].scores["rec"])
            return np.where(use, case.scores["oracle_recognition"], native)

        result[first] = dict(delta_pp_80=paired(cases, candidate, reference, orders), people={})
        for person in ("P04", "P10"):
            subset = [c for c in cases if c.cohort == "ret" and c.person == person]
            result[first]["people"][person] = score(subset, candidate, "ret") - score(subset, reference, "ret")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
