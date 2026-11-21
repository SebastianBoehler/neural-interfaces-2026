"""Capture the signed em_half true-recognition prefix on pinned Leonardo folds."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from capture import DATA_SHA, DEVICE, V, check, fixture_assets, load, open_fold
from returning_em import ARMS, patch
from returning_em_worker import fresh, modified
from tape import replay, tap


ROSTER = "ba_six"


def forced_fresh(person_index):
    def make(template):
        model = fresh(template)
        model.route.contrast.step = lambda *_: (person_index, 1.0, 1.0)
        model.route.identity.step = lambda *_: None
        return model

    return make


def delivery_labels(items):
    return np.asarray([item.label for item in items], dtype=np.int8)


def prefix(model, signals, rows, expected, person_index):
    native = tap(model, signals, rows, fresh, check, DEVICE)
    np.testing.assert_array_equal(delivery_labels(native), expected["native"])
    replay(model, native, "w25")
    flags = np.array([item.recognised for item in native])
    np.testing.assert_array_equal(flags, np.arange(len(rows)) >= expected["age"] - 1
                                  if expected["age"] > 0 else np.zeros(len(rows), dtype=bool))
    half_labels, _ = modified(model, native, "half")
    np.testing.assert_array_equal(half_labels, expected["half"])

    used = len(rows) - int(expected["age"] > 0)
    forced = tap(model, signals, rows[:used], forced_fresh(person_index), check, DEVICE)
    if not all(item.recognised and item.contrast[0] == person_index for item in forced):
        raise ValueError("true-identity replay did not take the personal path from delivery 1")
    replay(model, forced, "w25")
    labels, scores = modified(model, forced, "half")
    return native, forced, labels, scores


def smoke(pkg, receipt, data, signals, rows, expected, person_index, native, forced, labels, scores):
    with TemporaryDirectory() as temp:
        root = Path(temp)
        target = root / ROSTER
        target.mkdir()
        for path in pkg.iterdir():
            if path.name != "submission.py":
                (target / path.name).symlink_to(path.resolve())
        (target / "submission.py").write_text(patch((pkg / "submission.py").read_text(), ARMS["half"]))
        trial, _ = load(data, root, ROSTER, receipt)
        live = tap(trial, signals, rows, fresh, check, DEVICE)
        np.testing.assert_array_equal(delivery_labels(live), expected["half"])
        _, base_scores = modified(trial, native, "half")
        np.testing.assert_allclose([d.terminal for d in live], base_scores, rtol=0, atol=2e-5)
        live = tap(trial, signals, rows[:len(labels)], forced_fresh(person_index), check, DEVICE)
        np.testing.assert_array_equal(delivery_labels(live), labels)
        np.testing.assert_allclose([d.terminal for d in live], scores, rtol=0, atol=2e-5)

    short = min(10, len(labels))
    repeated = tap(trial, signals, rows[:short], forced_fresh(person_index), check, DEVICE)
    np.testing.assert_array_equal(delivery_labels(repeated), labels[:short])
    np.testing.assert_allclose([d.terminal for d in repeated], scores[:short], rtol=0, atol=2e-5)
    np.testing.assert_array_equal(delivery_labels(forced)[:short], labels[:short])
    print("PASS live half-weight patch, repeat/prefix and active true-identity route", flush=True)


def process(a):
    if not torch.cuda.is_available() or V.digest(a.data) != DATA_SHA:
        raise ValueError("CUDA and signed PROTEUS data required")
    data = V.D.load(a.data)
    signals = torch.from_numpy(data["X"])
    pkg, receipts = open_fold(a.fixtures, a.fold, "ba-confirm")
    receipt = receipts[ROSTER]
    person = a.fold.split("_")[1]
    names = fixture_assets(pkg / ROSTER, receipt)
    person_index = names.index(person)
    model, _ = load(data, pkg, ROSTER, receipt)
    if model.routeB.tail.em.nmin != 20:
        raise ValueError("unexpected half-weight SessionEM warm-up")

    block = "fit" if a.first < 71808 else "held"
    path = a.archive / block / f"{ROSTER}-{a.fold}-{a.first}.npz"
    payload = {"receipt": np.asarray(json.dumps(receipt, sort_keys=True))}
    with np.load(path, allow_pickle=False) as expected:
        if str(expected["receipt"]) != str(payload["receipt"]):
            raise ValueError("canonical archive used different assets")
        for order in range(a.first, a.first + 4):
            for visit in ("2", "3"):
                key = f"{order}_{visit}"
                stream = f"{person}|{visit}"
                rows = V.D.plans(data, order, [stream])[stream]
                pin_path = a.fixtures / "held" / a.fold / ROSTER / f"pred_{a.fold}_{order - order % 2}.npz"
                with np.load(pin_path, allow_pickle=False) as pinned:
                    if str(pinned["receipt"]) != str(payload["receipt"]):
                        raise ValueError("pinned labels used different assets")
                    np.testing.assert_array_equal(rows, pinned[f"rows_{key}"])
                    np.testing.assert_array_equal(expected[f"native_{key}"], pinned[f"labels_{key}"])
                np.testing.assert_array_equal(rows, expected[f"rows_{key}"])
                age = int(expected[f"rec_{key}"])
                cut = len(rows) if age == -1 else age
                if age != -1 and (not 1 < age <= len(rows) or str(expected[f"identity_{key}"]) != person):
                    raise ValueError(f"invalid genuine latch {a.fold}/{key}: {age}")
                window = {"age": age, **{arm: expected[f"{arm}_{key}"][:cut] for arm in ("native", "half")}}
                native, forced, labels, scores = prefix(model, signals, rows[:cut], window, person_index)
                if a.smoke:
                    smoke(pkg / ROSTER, receipt, data, signals, rows[:cut], window, person_index,
                          native, forced, labels, scores)
                    return
                payload.update({f"rows_{key}": rows[:len(labels)], f"oracle_{key}": labels,
                                f"native_gate_{key}": np.array([
                                    [(-1, np.nan, 0.0) if gate is None else gate
                                     for gate in (item.contrast, item.identity)] for item in native
                                ])})
                print(f"PASS {a.fold} {key}: {cut} pinned native and em_half deliveries, {len(labels)} oracle",
                      flush=True)
    dest = a.out / f"{a.fold}-{a.first}.npz"
    a.out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=a.out) as temp:
        target = Path(temp) / dest.name
        np.savez_compressed(target, **payload)
        target.rename(dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "fixtures", "archive", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("fold", choices=("ret_P04", "ret_P10"))
    parser.add_argument("first", type=int, choices=(71800, 71804, 71808, 71812))
    parser.add_argument("--smoke", action="store_true")
    process(parser.parse_args())


if __name__ == "__main__":
    main()
