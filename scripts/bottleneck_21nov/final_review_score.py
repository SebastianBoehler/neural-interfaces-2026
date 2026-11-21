"""Select on four fit orders and report the registered contrasts on four held orders."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np

from final_review_emissions import DOSES, balance_output, word_output
from oracles import freeze
from returning_em_report import load_cases
from tapes import joint, metadata, paired, score, stages
from terminal import check

ORDERS = {"fit": tuple(range(71800, 71804)), "held": tuple(range(71808, 71812))}
SPECS = ("balance", "word")
ROSTER = "ba_six"
DIAGNOSTIC_DOSE = 0.35


def stream_balance(raw, flags, strength):
    history = []
    return np.stack([balance_output(p, history, seen, strength) for p, seen in zip(raw, flags, strict=True)])


def stream_word(raw, team, manas, flags, strength):
    return np.stack([word_output(p, b, a, age, seen, strength)
                     for age, (p, b, a, seen) in enumerate(zip(raw, team, manas, flags, strict=True), 1)])


def emissions(case, legacy, doses):
    base = legacy.scores["half"]
    rec = legacy.scores["rec"]
    flags = (rec > 0) & (np.arange(len(base)) >= rec - 1)
    np.testing.assert_array_equal(case.rows, legacy.rows)
    np.testing.assert_array_equal(case.scores["recognised"], flags)
    raw = case.scores["terminal"]
    np.testing.assert_array_equal(raw.argmax(1)[~flags], base[~flags])
    compute = {
        "balance": (19, lambda n, s: stream_balance(raw[:n], flags[:n], s)),
        "word": (40, lambda n, s: stream_word(raw[:n], case.scores["team"][:n],
                                              case.scores["manas"][:n], flags[:n], s)),
    }
    out = {}
    for name, (warm, stream) in compute.items():
        inactive = (np.arange(len(base)) < warm) | flags
        for strength in doses[name]:
            q = stream(len(raw), strength)
            np.testing.assert_array_equal(q[inactive], raw[inactive])
            if strength == 0:
                np.testing.assert_array_equal(q, raw)
            np.testing.assert_array_equal(stream(67, strength), q[:67])
            out[name, strength] = np.where(flags, base, q.argmax(1))
            if strength == 0:
                np.testing.assert_array_equal(out[name, strength], base)
    return base, out


def summarize(cases, orders, predictions, name, strength):
    def reference(case):
        return predictions[case.key][0]

    def candidate(case):
        return predictions[case.key][1][name, strength]

    delta = paired(cases, candidate, reference, orders)
    by_cohort = {cohort: [c for c in cases if c.cohort == cohort] for cohort in ("new", "ret")}
    changed = {cohort: sum(int(np.count_nonzero(candidate(c) != reference(c))) for c in selected)
               for cohort, selected in by_cohort.items()}
    returning = {}
    for person in ("P04", "P10"):
        subset = [c for c in by_cohort["ret"] if c.person == person]
        returning[person] = score(subset, candidate, "ret") - score(subset, reference, "ret")
    point, low, _ = delta["J"]
    result = {"strength": strength, "delta_pp_80": delta, "changed": changed,
              "returning_people_pp": returning,
              "meets_numeric_rule": bool(point >= 0.15 and low > 0 and delta["new"][0] >= -0.10)}
    if name == "word":
        result["word_repairs_and_regressions"] = {
            cohort: [sum(int(np.count_nonzero((candidate(c) == 2) & (reference(c) != 2) & (c.y == 2)))
                         for c in selected),
                     sum(int(np.count_nonzero((candidate(c) == 2) & (reference(c) != 2) & (reference(c) == c.y)))
                         for c in selected)]
            for cohort, selected in by_cohort.items()
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=ORDERS)
    for name in ("data", "work", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--smoke", action="store_true", help="verify inputs, identity and prefix only")
    a = parser.parse_args()
    phase, orders = a.phase, ORDERS[a.phase]
    info = metadata(a.data)
    cases = stages(info, a.work, ROSTER, orders)
    check(cases, ROSTER)
    legacy = {c.key: c for c in load_cases(a.work, info, ROSTER, phase)}
    fit_file = a.out / "fit.json"
    doses = dict.fromkeys(SPECS, DOSES)
    if phase == "held":
        with fit_file.open("rb") as source:
            fit_sha = file_digest(source, "sha256").hexdigest()
        fit = json.loads(fit_file.read_text())
        if tuple(fit["orders"]) != ORDERS["fit"]:
            raise ValueError("wrong fit freeze")
        for name in SPECS:
            strength = fit["arms"][name]["selected"]
            if strength not in DOSES:
                raise ValueError("frozen dose outside the registered grid")
            doses[name] = tuple(dict.fromkeys((0.0, strength, DIAGNOSTIC_DOSE)))
    predictions = {c.key: emissions(c, legacy[c.key], doses) for c in cases}
    reference = lambda c: predictions[c.key][0]
    if a.smoke:
        for name in SPECS:
            if not any(np.any(v[1][name, doses[name][-1]] != v[0]) for v in predictions.values()):
                raise ValueError(f"{name} did not reach any fit decision")
        print(f"PASS {len(cases)} sessions: canonical labels, zero, activation, prefix and session reset")
        return
    a.out.mkdir(parents=True, exist_ok=True)
    baseline = joint(cases, reference)
    if phase == "fit":
        arms = {}
        for name in SPECS:
            values = [
                (s, joint(cases, lambda c: predictions[c.key][1][name, s]) - baseline)
                for s in doses[name]
            ]
            best = max(point for _, point in values)
            selected = next(s for s, point in values if point >= best - 1e-10)
            arms[name] = {"selected": selected, "fit_gain_pp": dict(values)}
        result = {"orders": orders, "baseline_50_50": baseline, "arms": arms}
        receipt = fit_file
    else:
        arms = {name: summarize(cases, orders, predictions, name, fit["arms"][name]["selected"])
                for name in SPECS}
        diagnostics = {name: summarize(cases, orders, predictions, name, DIAGNOSTIC_DOSE)
                       for name in SPECS if fit["arms"][name]["selected"] != DIAGNOSTIC_DOSE}
        result = {"orders": orders, "baseline_50_50": baseline, "fit_sha256": fit_sha,
                  "arms": arms, "active_diagnostics": diagnostics}
        receipt = a.out / "held.json"
    serialized = json.dumps(result, indent=2)
    freeze(receipt, result)
    print(serialized)


if __name__ == "__main__":
    main()
