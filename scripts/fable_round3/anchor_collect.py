"""Select the anchor dose on returning fit orders and confirm one choice."""
import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from anchor_worker import DOSES, RETURN_FOLDS
from decisions import load
from metrics import column, official_cells, paired, summarize
from settings import DATA_SHA, D, FIT, FOLDS, HELD, digest, streams
import collect as C


def attach_predictions(base, run, phase, doses, data, orders):
    cases, identities = load(base, "trace", data, orders)
    by_session = {(c.mode, c.person, c.order, c.visit): c for c in cases}
    needed = RETURN_FOLDS if phase == "fit" else FOLDS
    activations, banks = {}, {}
    keys = ("weights_sha", "base_weights_sha", "source_sha")
    for fold in needed:
        mode, person = fold.split("_")
        path = run / "banks" / f"{fold}.npz"
        banks[fold] = digest(path)
        with np.load(path, allow_pickle=False) as info:
            source = str(info["source_anchor_sha"])
            assert len(source) == 64
            for k in keys:
                assert str(info[k]) == identities[fold][k]
            roster = tuple(str(n) for n in info["names"])
        for i in range(0, len(orders), 2):
            chunk = orders[i:i + 2]
            with np.load(run / phase / f"{fold}_{chunk[0]}.npz", allow_pickle=False) as tape:
                assert tuple(tape["orders"]) == chunk and tuple(tape["doses"]) == doses
                assert str(tape["bank_sha"]) == banks[fold] and str(tape["source_anchor_sha"]) == source
                for k in keys:
                    assert str(tape[k]) == identities[fold][k]
                if phase == "held":
                    assert str(tape["selection_sha"]) == digest(run / "select.json")
                for order in chunk:
                    for stream, rows in D.plans(data, order, streams(data, fold)).items():
                        visit = stream.split("|")[1]
                        case = by_session[mode, person, order, visit]
                        key = f"{order}_{visit}"
                        np.testing.assert_array_equal(tape[f"rows_{key}"], rows)
                        for n0 in doses:
                            arm = f"n{n0}"
                            labels, age, j = (tape[f"{prefix}{arm}_{key}"] for prefix in ("", "age_", "person_"))
                            age, j = int(age), int(j)
                            assert labels.shape == rows.shape and np.isin(labels, (0, 1, 2)).all()
                            assert (age == j == -1) or (1 <= age <= len(rows) and 0 <= j < len(roster))
                            case.tape[arm] = labels
                            activations[f"{fold}/{order}/{visit}/{arm}"] = (age, None if j < 0 else roster[j])
    usable = cases if phase == "held" else [c for c in cases if c.mode == "ret"]
    return usable, banks, activations


def fit(cases):
    C.ORDERS = FIT
    seeds = np.random.default_rng(20261121).integers(len(FIT), size=(20_000, len(FIT)))
    base = official_cells(cases, column("lead"))
    assert len(base) == 8 * len(FIT)
    scores = {}
    for n0 in DOSES:
        trial = official_cells(cases, column(f"n{n0}"))
        assert trial.keys() == base.keys()
        delta = {k: trial[k] - base[k] for k in base}
        scores[n0], _ = C.by_cohort(delta, "ret", D.RETURNING, seeds, np.random.default_rng(71120))
    dose = min(DOSES, key=lambda n: (-round(scores[n]["mean"], 10), abs(n - 8), n))
    return dict(dose=dose, fit_returning=scores)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("select", "confirm"))
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    a = p.parse_args()
    if digest(a.data) != DATA_SHA:
        raise ValueError("unexpected PROTEUS export")
    data = D.load(a.data)
    if a.stage == "select":
        cases, hashes, activation = attach_predictions(a.base, a.run, "fit", DOSES, data, FIT)
        result = fit(cases)
    else:
        chosen = json.loads((a.run / "select.json").read_text())
        doses = (chosen["dose"],)
        cases, hashes, activation = attach_predictions(a.base, a.run, "held", doses, data, HELD)
        assert all(hashes[k] == v for k, v in chosen["bank_sha"].items())
        candidate, lead = column(f"n{doses[0]}"), column("lead")
        value = paired(cases, candidate, lead, HELD)
        early = lambda c: c.age <= 20
        result = dict(dose=doses[0], delta=value,
                      early=summarize(early, candidate, cases, HELD, early, lead),
                      go=value["returning"]["mean"] >= 0.2 and value["new"]["mean"] >= 0)
    result.update(bank_sha=hashes, activation=activation, data_sha=DATA_SHA)
    target = a.run / f"{a.stage}.json"
    with TemporaryDirectory(dir=a.run) as tmp:
        updated = Path(tmp) / target.name
        updated.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        os.link(updated, target)
    print(a.stage, digest(target), flush=True)


if __name__ == "__main__":
    main()
