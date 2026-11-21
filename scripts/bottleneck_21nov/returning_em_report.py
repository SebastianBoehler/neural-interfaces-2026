"""Freeze and verify the paired SessionEM decisions on disjoint pinned orders."""
import argparse
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np

from errors import fmt_ci, table
from oracles import freeze
from tapes import PEOPLE, ROSTERS, joint, metadata, paired, pinned, read_bank, score

ARMS = ("native", "half", "bypass")
ORDERS = {"fit": tuple(range(71800, 71808)), "held": tuple(range(71808, 71816))}


def predict(arm):
    return lambda case: case.scores[arm]


def load_cases(work, data, roster, phase):
    manifest = json.loads((work / "em_archive_manifest.json").read_text())
    canonical = {(x["fold"], key): age for x in manifest["active"]
                 if x["roster"] == roster and x["phase"] == phase
                 for key, age in x["recognition"].items()}

    def read(fold, order, key):
        chunk = order - order % 4
        path = work / "returning_em" / phase / f"{roster}-{fold}-{chunk}.npz"
        rows, labels, receipt = pinned(work / "leo", fold, roster, order, key)
        with np.load(path, allow_pickle=False) as tape:
            if str(tape["receipt"]) != receipt:
                raise ValueError(f"wrong frozen fixture: {path}")
            np.testing.assert_array_equal(rows, tape[f"rows_{key}"])
            np.testing.assert_array_equal(labels, tape[f"native_{key}"])
            scores = {arm: tape[f"{arm}_{key}"] for arm in ARMS}
            age, identity = int(tape[f"rec_{key}"]), str(tape[f"identity_{key}"])
            if not (-1 == age and not identity or 1 <= age <= len(rows) and identity.startswith("P")):
                raise ValueError("invalid recognition receipt")
            if age != canonical.get((fold, key), -1):
                raise ValueError(f"native recognition does not match canonical age: {fold}/{key}")
            prefix = age - 1 if age > 0 else len(rows)
            for arm in ARMS:
                np.testing.assert_array_equal(labels[:prefix], scores[arm][:prefix], err_msg=f"{fold}/{key}/{arm}")
            scores.update(rec=np.full(len(rows), age),
                          identity=np.full(len(rows), int(identity[1:]) if identity else 0))
        return rows, scores
    return read_bank(data, ORDERS[phase], read)


def fit(cases):
    values = {arm: joint(cases, predict(arm)) for arm in ARMS}
    best = max(values.values())
    selected = next(arm for arm, value in values.items() if value >= best - 1e-10)
    return {"selected": selected, "gains_pp": {arm: v - values["native"] for arm, v in values.items()},
            "misrecognitions": misrecognitions(cases)}


def misrecognitions(cases):
    false = []
    for case in cases:
        age, identity = int(case.scores["rec"][0]), int(case.scores["identity"][0])
        if age > 0 and (case.cohort == "new" or identity != int(case.person[1:])):
            false.append(dict(cohort=case.cohort, person=case.person, visit=case.visit, order=case.order, age=age,
                              identity=f"P{identity:02d}", changes={arm: int(np.count_nonzero(
                                  case.scores[arm] != case.scores["native"])) for arm in ARMS[1:]}))
    return false


def held(cases):
    results = {}
    for arm in ARMS:
        result = paired(cases, predict(arm), predict("native"), ORDERS["held"])
        people = {}
        for person in PEOPLE["ret"]:
            selected = [c for c in cases if c.person == person]
            people[person] = score(selected, predict(arm), "ret") - score(selected, predict("native"), "ret")
        point, low, _ = result["J"]
        results[arm] = {"delta": result, "returning_people": people, "pass": bool(point >= .15 and low > 0)}
    return {"arms": results, "misrecognitions": misrecognitions(cases)}


def markdown(results):
    rows, text = [], []
    for roster in ROSTERS:
        r = results["rosters"][roster]
        for arm in ARMS[1:]:
            a = r["arms"][arm]
            delta = a["delta"]
            flag = "frozen · passes" if r["selected"] == arm and a["pass"] else (
                   "frozen" if r["selected"] == arm else "secondary")
            people = a["returning_people"]
            rows.append((roster, arm, f"{r['fit_gains_pp'][arm]:+.2f}",
                         *(fmt_ci(delta[c]) for c in ("new", "ret", "J")),
                         f"{people['P04']:+.2f}/{people['P10']:+.2f}", flag))
        if r["selected"] == "native":
            text.append(f"{roster} froze native weight 1.0.")
        if not r["fit_misrecognitions"] and not r["misrecognitions"]:
            text.append(f"{roster} had zero misrecognitions on both order blocks.")
        for phase, events in (("fit", r["fit_misrecognitions"]), ("held-out", r["misrecognitions"])):
            for f in events:
                counts = f["changes"]
                text.append(f"{roster} misrecognised {f['cohort']} {f['person']}/{f['visit']} as {f['identity']} "
                            f"on {phase} order {f['order']} at delivery {f['age']}. "
                            f"Half/bypass changed {counts['half']}/{counts['bypass']} labels.")
    head = ["Stack", "EM arm", "Fit J Δ", "New Δ [80%]", "Ret. Δ [80%]", "50/50 Δ [80%]", "P04/P10 Δ", "Role"]
    return "\n\n".join([table(head, rows), *text, f"Fit freeze SHA-256 `{results['fit_sha256']}`."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=ORDERS)
    parser.add_argument("work", type=Path)
    parser.add_argument("data", type=Path)
    args = parser.parse_args()
    phase, work = args.phase, args.work
    data = metadata(args.data)
    frozen = work / "em-fit.json"
    if phase == "fit":
        result = {"orders": ORDERS["fit"],
                  "rosters": {r: fit(load_cases(work, data, r, phase)) for r in ROSTERS}}
        freeze(frozen, result)
        print(json.dumps(result, indent=2))
        return
    with frozen.open("rb") as f:
        fit_hash = file_digest(f, "sha256").hexdigest()
    fit_result = json.loads(frozen.read_text())
    if tuple(fit_result["orders"]) != ORDERS["fit"]:
        raise ValueError("invalid fit split")
    rosters = {}
    for r in ROSTERS:
        a = fit_result["rosters"][r]
        stats = held(load_cases(work, data, r, phase))
        rosters[r] = {**stats, "selected": a["selected"], "fit_gains_pp": a["gains_pp"],
                      "fit_misrecognitions": a["misrecognitions"],
                      "pass": stats["arms"][a["selected"]]["pass"]}
    result = {"orders": ORDERS[phase], "fit_sha256": fit_hash, "rosters": rosters}
    freeze(work / "em-held.json", result)
    print(markdown(result))


if __name__ == "__main__":
    main()
