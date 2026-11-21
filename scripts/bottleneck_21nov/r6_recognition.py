"""Score the 16-order early-recognition oracle and impostor crossing diagnostic."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from oracles import freeze
from r6_recognition_stat import ROSTER, crossings, digest, reconstruct, thresholds
from returning_em_report import ORDERS, load_cases, misrecognitions
from tapes import joint, metadata, paired, pinned, score, stages
from terminal import check


TIMES = (1, 3, 5, 8, 12)
CAPTURE_ORDERS = (71800, 71801, 71802, 71803, 71808, 71809, 71810, 71811)


def load_oracle(work, out, cases, stored):
    oracle, witnesses, paths = {}, {}, set()
    for case in cases:
        if case.cohort != "ret":
            continue
        fold, key = f"ret_{case.person}", f"{case.order}_{case.visit}"
        age = int(case.scores["rec"][0])
        n = len(case.rows) if age == -1 else age - 1
        _, _, receipt = pinned(work / "leo", fold, ROSTER, case.order, key)
        path = out / f"{fold}-{case.order - case.order % 4}.npz"
        with np.load(path, allow_pickle=False) as capture:
            if str(capture["receipt"]) != receipt:
                raise ValueError(f"unexpected new prefix fixture: {path}")
            np.testing.assert_array_equal(capture[f"rows_{key}"], case.rows[:n])
            q = capture[f"oracle_{key}"]
            witnesses[case.key] = capture[f"native_gate_{key}"]
        if q.shape != (n,) or not np.isin(q, (0, 1, 2)).all():
            raise ValueError(f"unexpected oracle decision shape or value: {case.key}")
        if case.key in stored:
            reference = stored[case.key]
            np.testing.assert_array_equal(reference.rows, case.rows)
            np.testing.assert_array_equal(reference.scores["recognised"],
                                          (age > 0) & (np.arange(len(case.rows)) >= age - 1))
            # SessionEM nmin=20. The old oracle uses weight 1 on active emissions.
            np.testing.assert_array_equal(reference.scores["oracle_recognition"][:min(n, 20)], q[:20])
        oracle[case.key] = q
        paths.add(path)
    return oracle, witnesses, paths


def base(case):
    return case.scores["half"]


def splice(case, oracle, first):
    current = base(case)
    if case.cohort != "ret" or first is None:
        return current
    replacement = oracle[case.key]
    if len(replacement) < first:
        return current
    return np.concatenate((current[:first - 1], replacement[first - 1:], current[len(replacement):]))


def report(cases, firsts, oracle, orders):
    def trial(case):
        return splice(case, oracle, firsts.get(case.key))

    people = {}
    for person in ("P04", "P10"):
        subset = [case for case in cases if case.person == person and case.cohort == "ret"]
        people[person] = score(subset, trial, "ret") - score(subset, base, "ret")
    return {"delta_pp_80": paired(cases, trial, base, orders), "returning_people_pp": people}


def histogram(cases, value):
    return {person: dict(sorted(Counter(value(c) for c in cases if c.person == person).items()))
            for person in ("P04", "P10")}


def one_crossing_rule(cases, traces, oracle, orders, minima, threshold, fit_threshold):
    pooled = crossings(traces, minima, threshold)
    if any(key[0] == "new" for key in pooled):
        raise ValueError("strict pooled threshold allowed an impostor crossing")
    returning = [c for c in cases if c.cohort == "ret"]
    correct, wrong = {}, []
    for c in returning:
        if c.key in pooled:
            age, person = pooled[c.key]
            if int(c.person[1:]) == person:
                correct[c.key] = age
            else:
                wrong.append((*c.key, age, person))
    fit_policy = crossings(traces, minima, fit_threshold)
    fit_to_held = [(*key, *value) for key, value in fit_policy.items() if key[0] == "new" and key[1] >= 71808]
    earlier = [c for c in returning if c.key in correct and
               (int(c.scores["rec"][0]) < 0 or correct[c.key] < int(c.scores["rec"][0]))]
    limits = ({"pooled_strict_threshold": float(threshold), "fit_strict_threshold": float(fit_threshold)}
              if np.ndim(threshold) == 0 else {})
    return {**limits,
            "earliest_correct": min(correct.values(), default=None),
            "genuine_crossings": histogram(returning, lambda c: correct.get(c.key, -1)),
            "genuine_by_5": sum(age <= 5 for age in correct.values()),
            "genuine_earlier_than_native": histogram(earlier, lambda c: correct[c.key]),
            "wrong_genuine": wrong, "fit_threshold_held_impostor_crossings": fit_to_held,
            "splice": report(cases, correct, oracle, orders)}


def crossing_report(cases, traces, oracle, orders):
    minima, policies = thresholds(traces, cases)
    return {name: one_crossing_rule(cases, traces, oracle, orders, minima, *limits)
            for name, limits in policies.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "work", "prefix", "fixtures", "archive", "out"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    data = metadata(a.data)
    by_phase = {phase: load_cases(a.work, data, ROSTER, phase) for phase in ORDERS}
    cases = by_phase["fit"] + by_phase["held"]
    orders = ORDERS["fit"] + ORDERS["held"]
    old = stages(data, a.work, ROSTER, CAPTURE_ORDERS)
    check(old, ROSTER)
    oracle, witnesses, paths = load_oracle(a.work, a.prefix, cases, {c.key: c for c in old})
    null = lambda c: splice(c, oracle, len(c.rows) + 1)
    for case in cases:
        np.testing.assert_array_equal(null(case), base(case))
    if paired(cases, null, base, orders)["J"] != (0.0, 0.0, 0.0):
        raise ValueError("nonzero change with inactive splice")
    traces, verified = reconstruct(a.data, a.work, a.fixtures, a.archive, cases, witnesses)
    result = {"orders": orders, "base_50_50": float(joint(cases, base)),
              "prefix_sha256": {path.name: digest(path) for path in sorted(paths)},
              "returning_latches": histogram([c for c in cases if c.cohort == "ret"],
                                             lambda c: int(c.scores["rec"][0])),
              "new_misrecognitions": {phase: misrecognitions(batch) for phase, batch in by_phase.items()},
              "native_identity_latch_agrees": sum(
                  verified[c.key]["identity_first"] == (int(c.scores["rec"][0]) if c.scores["rec"][0] > 0 else None)
                  for c in cases),
              "curves": {}, "crossing": crossing_report(cases, traces, oracle, orders)}
    for first in TIMES:
        firsts = {key: first for key in oracle}
        result["curves"][first] = {
            "all": report(cases, firsts, oracle, orders),
            **{phase: report(batch, firsts, oracle, ORDERS[phase]) for phase, batch in by_phase.items()}
        }
    result["stage_2_released"] = bool(result["curves"][5]["all"]["delta_pp_80"]["J"][0] >= 0.30)
    freeze(a.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
