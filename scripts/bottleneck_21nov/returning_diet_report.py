"""Collect paired diet results using authenticated em_half controls."""
import argparse
from collections import Counter
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np

from oracles import freeze
from returning_diet import ARMS, PARENT_SHA
from returning_em_report import ORDERS, load_cases
from tapes import joint, metadata, paired, pinned, score

ROSTER = "ba_six"
REFERENCE = "half"


def digest(path):
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def load_bank(work, captures, data, phase):
    cases = load_cases(work, data, ROSTER, phase)
    files = {}
    for case in cases:
        saved = case.scores[REFERENCE]
        age = int(case.scores["rec"][0])
        if age == -1:
            case.scores.update({arm: saved for arm in ARMS})
            continue
        fold, key = f"{case.cohort}_{case.person}", f"{case.order}_{case.visit}"
        path = captures / phase / f"{fold}-{case.order - case.order % 4}.npz"
        if path not in files:
            with np.load(path, allow_pickle=False) as tape:
                files[path] = dict(tape)
        tape = files[path]
        _, _, receipt = pinned(work / "leo", fold, ROSTER, case.order, key)
        if str(tape["receipt"]) != receipt:
            raise ValueError(f"unexpected fixture in {path}")
        np.testing.assert_array_equal(tape[f"rows_{key}"], case.rows)
        np.testing.assert_array_equal(tape[f"half_{key}"], saved)
        for arm in ARMS:
            pred = tape[f"{arm}_{key}"]
            if pred.shape != saved.shape or pred.dtype != np.int8 or not np.isin(pred, (0, 1, 2)).all():
                raise ValueError(f"invalid {arm} labels in {path}")
            np.testing.assert_array_equal(pred[:age - 1], saved[:age - 1])
            case.scores[arm] = pred
    return cases, {f"{phase}/{p.name}": digest(p) for p in files}


def predict(arm):
    return lambda case: case.scores[arm]


def levels(cases, arm):
    read = predict(arm)
    return dict(new=score(cases, read, "new"), ret=score(cases, read, "ret"), J=joint(cases, read))


def person_delta(cases, orders, arm, person, *, reference=REFERENCE):
    eligible = [c for c in cases if c.cohort == "ret" and c.person == person]
    candidate, parent = predict(arm), predict(reference)
    differences = []
    for order in orders:
        block = [c for c in eligible if c.order == order]
        differences.append(score(block, candidate, "ret") - score(block, parent, "ret"))
    indices = np.random.default_rng(20261121).integers(len(orders), size=(20_000, len(orders)))
    draws = np.asarray(differences)[indices].mean(1)
    return (np.mean(differences), *np.quantile(draws, (0.1, 0.9)))


def report(cases, orders, *, arms=ARMS, nested_orders=False):
    result = {REFERENCE: levels(cases, REFERENCE)}
    misrecognised = []
    coverage = Counter()
    for case in cases:
        ref = case.scores[REFERENCE]
        age, identity = (int(case.scores[k][0]) for k in ("rec", "identity"))
        coverage[f"{case.cohort}_windows"] += len(ref)
        if age < 0:
            continue
        coverage[f"{case.cohort}_recognised_streams"] += 1
        if case.cohort == "ret" and identity == int(case.person[1:]):
            continue
        misrecognised.append(dict(person=case.person, visit=case.visit, order=case.order, age=age,
                                  identity=f"P{identity:02d}", cohort=case.cohort,
                                  changes={a: int(np.count_nonzero(case.scores[a] != ref)) for a in arms}))
    for arm in arms:
        delta = paired(cases, predict(arm), predict(REFERENCE), orders, nested_orders=nested_orders)
        people = {p: person_delta(cases, orders, arm, p) for p in ("P04", "P10")}
        changes = {mode: sum(int(np.count_nonzero(c.scores[arm] != c.scores[REFERENCE]))
                             for c in cases if c.cohort == mode) for mode in ("new", "ret")}
        result[arm] = dict(BA=levels(cases, arm), delta=delta, people=people, changed_labels=changes,
                           passes_rule=bool(delta["ret"][0] >= 0.30 and people["P10"][0] >= -0.20))
    return dict(arms=result, misrecognised=misrecognised, coverage=coverage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=ORDERS)
    for name in ("archive", "captures", "data", "out"):
        parser.add_argument(name, type=Path)
    a = parser.parse_args()
    orders = ORDERS[a.phase]
    protocol = dict(parent_sha256=PARENT_SHA, doses=ARMS, held=list(ORDERS["held"]))
    fit = a.out / "fit.json"
    result = {"protocol": protocol, "orders": orders}
    if a.phase == "held":
        frozen = json.loads(fit.read_text())
        if frozen["protocol"] != protocol or frozen["orders"] != list(ORDERS["fit"]):
            raise ValueError("fit report used different protocol")
        result["fit_sha256"] = digest(fit)
    data = metadata(a.data)
    bank, files = load_bank(a.archive, a.captures, data, a.phase)
    result.update(report(bank, orders), captures_sha256=files,
                  recognition_manifest_sha256=digest(a.archive / "em_archive_manifest.json"))
    if a.phase == "held":
        result["promote"] = next((arm for arm in ARMS if result["arms"][arm]["passes_rule"]), None)
    a.out.mkdir(parents=True, exist_ok=True)
    freeze(a.out / f"{a.phase}.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
