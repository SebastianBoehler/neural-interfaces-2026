"""Collect MAIN-versus-quarter scores on frozen fit and held delivery orders."""
import argparse
import json
from pathlib import Path

import numpy as np

from manas_head_reachability import ZIP_SHA, digest
from returning_diet_report import load_bank, predict
from manas_head_gate import PEOPLE
from returning_em_report import ORDERS
from tapes import joint, metadata, paired, pinned, score


def collect(a):
    orders = ORDERS[a.phase]
    cases, diet_files = load_bank(a.archive, a.diet, metadata(a.data), a.phase)
    gate = json.loads((a.gate / f"{a.phase}.json").read_text())
    if not gate["exact_folds"] or gate["orders"] != list(orders):
        raise ValueError("expected an exact gate audit on the same orders")
    files, misrecognitions = {}, []
    counts = {"new": 0, "ret": 0}
    for case in cases:
        ref = case.scores["quarter"]
        age, identity = (int(case.scores[k][0]) for k in ("rec", "identity"))
        if age < 0:
            case.scores["main"] = ref
            continue
        fold = f"{case.cohort}_{case.person}"
        key = f"{case.order}_{case.visit}"
        path = a.captures / a.phase / f"{fold}-{case.order - case.order % 4}.npz"
        if path not in files:
            with np.load(path, allow_pickle=False) as tape:
                files[path] = dict(tape)
        item = files[path]
        _, _, receipt = pinned(a.archive / "leo", fold, "ba_six", case.order, key)
        if str(item["receipt"]) != receipt:
            raise ValueError(f"wrong fold receipt in {path}")
        np.testing.assert_array_equal(item[f"rows_{key}"], case.rows)
        np.testing.assert_array_equal(item[f"quarter_{key}"], ref)
        pred, head = item[f"main_{key}"], item[f"native_person_{key}"]
        if pred.shape != ref.shape or pred.dtype != np.int8 or not np.isin(pred, (0, 1, 2)).all():
            raise ValueError(f"invalid MAIN decisions: {path}/{key}")
        np.testing.assert_array_equal(pred[:age - 1], ref[:age - 1])
        native_age, native_name = gate["streams"][fold][key]["native"]
        slots = [int(person[1:]) for person in PEOPLE]
        if case.cohort == "new":
            slots.remove(int(case.person[1:]))
        expected = np.full(len(ref), -1)
        if native_age > 0:
            expected[native_age - 1:] = slots.index(int(native_name[1:]))
        np.testing.assert_array_equal(head, expected, err_msg=f"{fold}/{key}")
        expected[:] = -1
        expected[age - 1:] = slots.index(identity)
        np.testing.assert_array_equal(item[f"main_person_{key}"], expected)
        changed = int(np.count_nonzero(pred != ref))
        counts[case.cohort] += changed
        if case.cohort == "new" or identity != int(case.person[1:]):
            misrecognitions.append(dict(fold=fold, key=key, age=age, identity=f"P{identity:02d}", changed=changed))
        case.scores["main"] = pred
    reference, main = predict("quarter"), predict("main")
    delta = paired(cases, main, reference, orders)
    metrics = {arm: dict(new=score(cases, pred, "new"), ret=score(cases, pred, "ret"), J=joint(cases, pred))
               for arm, pred in (("quarter", reference), ("main", main))}
    returning = {}
    for person in ("P04", "P10"):
        block = [c for c in cases if c.person == person and c.cohort == "ret"]
        values = np.array([score([c for c in block if c.order == o], main, "ret") -
                           score([c for c in block if c.order == o], reference, "ret") for o in orders])
        indices = np.random.default_rng(20261121).integers(len(orders), size=(20_000, len(orders)))
        returning[person] = (values.mean(), *np.quantile(values[indices].mean(1), (0.1, 0.9)))
    result = dict(parent_sha256=ZIP_SHA, orders=orders, BA=metrics, delta_pp=delta,
                  returning_people_pp=returning, changed_labels=counts, misrecognitions=misrecognitions,
                  captures_sha256={p.name: digest(p) for p in files},
                  gate_sha256=digest(a.gate / f"{a.phase}.json"), diet_sha256=diet_files,
                  promote=bool(delta["J"][0] >= 0.15 and delta["J"][1] > 0 and delta["new"][0] >= -0.05))
    if a.phase == "held":
        fit = a.out / "fit.json"
        frozen = json.loads(fit.read_text())
        if frozen["orders"] != list(ORDERS["fit"]) or frozen["parent_sha256"] != ZIP_SHA:
            raise ValueError("different fit protocol")
        result["fit_sha256"] = digest(fit)
    return cases, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=ORDERS)
    for name in ("data", "archive", "diet", "gate", "captures", "out"):
        parser.add_argument(name, type=Path)
    a = parser.parse_args()
    _, result = collect(a)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.phase}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
