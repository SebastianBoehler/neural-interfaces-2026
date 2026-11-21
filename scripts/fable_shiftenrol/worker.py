"""Smoke or replay the registered returning enrolment contrast."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from fixtures import (D, DATA_SHA, HELD, PINNED_ORDER, PRECHECK, digest, enrol, load, pinned,
                      read_weights, replay, stage, stage_weights)
from fit import fit_person, half


def smoke(a, data, signals, ref, receipt, work):
    person = a.fold.split("_")[1]
    stream = f"{person}|2"
    rows = D.plans(data, PINNED_ORDER, [stream])[stream]
    baseline = load(ref, data)
    labels, meta = replay(baseline, signals, rows)
    np.testing.assert_array_equal(labels, pinned(a.inputs, a.fold, "nine25", stream, rows, receipt["weights_sha"]))
    noop = load(stage_weights(ref, work / "zero", enrol(read_weights(ref / "weights.npz"), {})), data)
    same, _ = replay(noop, signals, rows)
    np.testing.assert_array_equal(same, labels)
    if a.fold.startswith("ret_"):
        source, _ = half(data, person, "2")
        z = read_weights(ref / "weights.npz")
        options = fit_person(data, {"2": source}, person, z)
        row = list(z["route_names"]).index(person)
        if any(np.array_equal(values, z[key][row]) for key, values in options.items()):
            raise ValueError("same-day enrolment left a personalized array unmodified")
        candidate = load(stage_weights(ref, work / "same", enrol(z, {person: options})), data)
        _, same_meta = replay(candidate, signals, rows[np.isin(rows, source, invert=True)][:72])
        if meta["routed"] == 0 or same_meta["own"] == 0:
            raise ValueError(f"personal route did not engage: {meta}, {same_meta}")
    print(f"PASS pinned and weights-only zero {a.fold}: {len(rows)} windows", flush=True)


def augmented_model(a, data, signals, ref, work):
    person = a.fold.split("_")[1]
    z = read_weights(ref / "weights.npz")
    source = {"1": np.flatnonzero(data["stream"] == f"{person}|1")}
    original = fit_person(data, source, person, z)
    index = list(z["route_names"]).index(person)
    for key, value in original.items():
        if key == "own_csp":
            # Eigenvector sign is arbitrary. OwnStage consumes only projected powers.
            actual, expected = np.abs(value), np.abs(z[key][index])
        else:
            actual, expected = value, z[key][index]
        np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=1e-4, err_msg=key)
    fitted = fit_person(data, source, person, z, augmented=True)
    if any(np.array_equal(fitted[key], values) for key, values in original.items()):
        raise ValueError("augmentation left a personal array unchanged")
    target = stage_weights(ref, work / "augmented", enrol(z, {person: fitted}))
    rows = D.plans(data, PINNED_ORDER, [f"{person}|2"])[f"{person}|2"][:72]
    _, meta = replay(load(target, data), signals, rows)
    if meta["own"] == 0:
        raise ValueError("augmented enrolment failed to route the smoke prefix")
    with a.out.open("xb") as f:
        np.savez(f, **fitted, fold=a.fold, base_weights_sha=digest(ref / "weights.npz"), source_rows=source["1"])
    print(f"PASS original fit parity and augmented live prefix for {a.fold}", flush=True)


def compare(a, data, signals, ref, receipt, work):
    person = a.fold.split("_")[1]
    if person not in D.RETURNING:
        raise ValueError("returning comparison requires P04 or P10")
    z, baseline = read_weights(ref / "weights.npz"), load(ref, data)
    cases = []
    for visit in a.visits:
        if a.phase == "day":
            want, hold = half(data, person, visit)
            source, orders = {visit: want}, PRECHECK
        else:
            source = {"1": np.flatnonzero(data["stream"] == f"{person}|1")}
            hold = np.flatnonzero(data["stream"] == f"{person}|{visit}")
            orders = HELD
        if any(np.intersect1d(rows, hold).size for rows in source.values()):
            raise ValueError("enrolment overlaps the tested rows")
        target = work / f"candidate_{visit}"
        if a.phase == "held":
            fitted = read_weights(a.augmented)
            if (str(fitted.pop("fold")) != a.fold
                    or str(fitted.pop("base_weights_sha")) != receipt["base_weights_sha"]):
                raise ValueError("augmentation was trained on a different fold")
            np.testing.assert_array_equal(fitted.pop("source_rows"), source["1"])
        else:
            fitted = fit_person(data, source, person, z)
        fitted_weights = enrol(z, {person: fitted})
        model = load(stage_weights(ref, target, fitted_weights), data)
        for order in orders:
            stream = f"{person}|{visit}"
            rows = D.plans(data, order, [stream])[stream]
            rows = rows[np.isin(rows, hold)]
            base, base_meta = replay(baseline, signals, rows)
            new, new_meta = replay(model, signals, rows)
            cases.append(dict(visit=visit, order=order, rows=rows.tolist(), base=base.tolist(), new=new.tolist(),
                              base_meta=base_meta, new_meta=new_meta, source={k: v.tolist() for k, v in source.items()},
                              weights_sha=digest(target / "weights.npz")))
            print(a.phase, a.fold, visit, order, len(rows), base_meta, new_meta, flush=True)
        del model
    with a.out.open("x") as file:
        json.dump(dict(fold=a.fold, phase=a.phase, source=receipt, data_sha=DATA_SHA, cases=cases), file)
        file.write("\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("smoke", "day", "aug-smoke", "held"))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--lead", type=Path, required=True)
    p.add_argument("--fold", required=True)
    p.add_argument("--visits", nargs="+", choices=("2", "3"), default=("2", "3"))
    p.add_argument("--out", type=Path)
    p.add_argument("--augmented", type=Path)
    a = p.parse_args()
    if digest(a.data) != DATA_SHA or not torch.cuda.is_available():
        raise ValueError("expected the signed public export and a CUDA worker")
    torch.backends.cudnn.benchmark = False
    data = D.load(a.data)
    signals = torch.from_numpy(data["X"])
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        ref, receipt = stage(data, a.fold, a.inputs, a.lead, work)
        if a.phase == "smoke":
            smoke(a, data, signals, ref, receipt, work)
        elif a.phase == "aug-smoke":
            augmented_model(a, data, signals, ref, work)
        else:
            compare(a, data, signals, ref, receipt, work)


if __name__ == "__main__":
    main()
