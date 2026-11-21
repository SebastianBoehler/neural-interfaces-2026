"""Run and collect paired arm c extensions on authenticated PROTEUS folds."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

import arm_c_confirm as control
import arm_c_sprint_policy as sprint
from diet_decomposition_patch import ARMS as DIET_ARMS, patch as component_patch
from returning_diet_report import load_bank

ORDERS = {"held": tuple(range(71808, 71816)), "fresh": control.ORDERS}
FOLDS = control.FOLDS
ROSTER = control.ROSTER
MODES = ("held", "fresh")


@lru_cache(maxsize=64)
def read(path):
    with np.load(path, allow_pickle=False) as file:
        return dict(file)


def archive_path(history, scheme, fold, order):
    if order in ORDERS["fresh"]:
        return history / "gpt-arm-c-confirm-21nov/out" / scheme / f"{fold}-{order}.npz"
    if scheme == "full":
        return history / "diet-decomposition-21nov/out/held" / f"{fold}-{order - order % 4}.npz"
    return None


def smoke_live(model, package, data, signals, rows, items, receipt, reference):
    prefix = rows[:84]
    c_source = component_patch(control.patch_half((package / "submission.py").read_text(), 0.5), DIET_ARMS["c"])
    for arm in ("c", *sprint.ARMS):
        text = c_source if arm == "c" else sprint.patch(c_source, arm)
        result = sprint.checked_replay(model, items, reference, arm)
        for full, short in zip(result, sprint.intervene(model, items[:len(prefix)], arm), strict=True):
            np.testing.assert_array_equal(full[:len(prefix)], short, err_msg=f"{arm} prefix causality")
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            control.stage(folder / ROSTER, package, text)
            live, _ = control.load(data, folder, ROSTER, receipt)

            def check(m, n):
                control.check(m, n)
                tail = m.routeB.tail
                assert m.cf.t == m.cp.t == len(tail.em.A) == len(tail.tsem.U) == n
                assert tail.em.state is not None and tail.tsem.state is not None

            seen = control.tap(live, signals, prefix, control.fresh, check, control.DEVICE)
        np.testing.assert_array_equal([i.recognised for i in seen], [i.recognised for i in items[:len(prefix)]])
        for key, expected in zip(("label", "terminal", "route_out"), result, strict=True):
            recorded = np.asarray([getattr(i, key) for i in seen])
            if key == "label":
                np.testing.assert_array_equal(recorded, expected[:len(prefix)], err_msg=arm)
            else:
                np.testing.assert_allclose(recorded, expected[:len(prefix)], rtol=0, atol=2e-5, err_msg=arm)
        print(f"PASS {arm}: live full tail and causal prefix", flush=True)


def score(args):
    if not torch.cuda.is_available() or control.V.digest(args.data) != control.DATA_SHA:
        raise ValueError("CUDA and signed PROTEUS data required")
    smoke = args.mode == "smoke"
    order_index, fold_index = divmod(args.task, len(FOLDS))
    order = ORDERS["held" if smoke else args.mode][order_index]
    fold = FOLDS[fold_index]
    data = control.V.D.load(args.data)
    signals = torch.from_numpy(data["X"])
    package, receipts = control.open_fold(args.fixtures, fold, "ba-confirm")
    receipt = receipts[ROSTER]
    people = control.fixture_assets(package / ROSTER, receipt)
    model, _ = control.load(data, package, ROSTER, receipt)
    streams = control.sessions(data, fold)
    if smoke:
        streams = streams[-1:]
    for scheme in control.SCHEMES:
        if args.mode == "held" and scheme == "full" and fold.startswith("new_"):
            continue
        archived = archive_path(args.history, scheme, fold, order)
        pinned = read(archived) if archived else None
        payload = {"receipt": np.asarray(json.dumps(receipt, sort_keys=True))}
        if pinned is not None and str(pinned["receipt"]) != str(payload["receipt"]):
            raise ValueError("arm c archive belongs to a different fold")
        for stream, rows in control.V.D.plans(data, order, streams, scheme=scheme).items():
            key = f"{order}_{stream.split('|')[1]}"
            items = control.taped(model, signals, rows)
            native = np.asarray([item.label for item in items], dtype=np.int8)
            control.replay(model, items, "w25")
            flags = np.asarray([item.recognised for item in items])
            age, identity = -1, 0
            outputs = {arm: native for arm in ("c", *sprint.ARMS)}
            activity = dict.fromkeys(sprint.ARMS, 0.0)
            if flags.any():
                i = int(flags.argmax())
                gate = control.top_gate(items[i].contrast, items[i].identity)
                if gate is None or gate[2] <= 0 or not flags[i:].all():
                    raise ValueError(f"invalid recognition latch: {fold}/{order}/{stream}")
                age, identity = i + 1, int(people[gate[0]][1:])
                reference = sprint.intervene(model, items, "c")
                np.testing.assert_array_equal(reference[0][~flags], native[~flags])
                outputs["c"] = reference[0]
                for arm in sprint.ARMS:
                    labels, terminal, route = sprint.checked_replay(model, items, reference, arm)
                    outputs[arm] = labels
                    activity[arm] = float(np.max(np.abs(route - reference[2])))
                if smoke:
                    smoke_live(model, package / ROSTER, data, signals, rows, items, receipt, reference)
            elif smoke:
                raise ValueError(f"smoke must exercise recognition: {scheme}/{fold}/{stream}")
            if pinned is not None:
                np.testing.assert_array_equal(rows, pinned[f"rows_{key}"], err_msg=f"{archived}/{key}")
                np.testing.assert_array_equal(outputs["c"], pinned[f"c_{key}"], err_msg=f"{archived}/{key}")
                if f"rec_{key}" in pinned:
                    assert age == int(pinned[f"rec_{key}"]) and identity == int(pinned[f"identity_{key}"])
            payload.update({f"rows_{key}": rows, f"rec_{key}": np.asarray(age), f"identity_{key}": np.asarray(identity)})
            payload.update({f"{arm}_{key}": labels for arm, labels in outputs.items()})
            payload.update({f"activity_{arm}_{key}": np.asarray(value) for arm, value in activity.items()})
            print(f"PASS {scheme} {fold}/{key}: {len(rows)} rows, rec {age}, identity {identity}, activity {activity}",
                  flush=True)
        if not smoke:
            folder = args.out / args.mode / scheme
            with TemporaryDirectory(dir=folder) as tmp:
                path = Path(tmp) / f"{fold}-{order}.npz"
                np.savez_compressed(path, **payload)
                path.rename(folder / path.name)


def collect(args):
    data = control.metadata(args.data)
    with np.load(args.data, allow_pickle=False) as source:
        data.update({key: source[key] for key in ("run", "onset")})
    streams = np.unique(data["stream"])
    receipts = {fold: json.loads((args.fixtures / "fixtures" / fold / "receipt.json").read_text())[ROSTER]
                for fold in FOLDS}
    history = args.history
    diet_dir = history / "diet-decomposition-21nov/out"
    pins_root = args.out.parent / "held-pins"
    manifest = pins_root / "em_archive_manifest.json"
    if control.digest(manifest) != "c4bbc41808d296aa4e2ee18151bf9680ed46f465b83462e94a6be632bed3dd3e":
        raise ValueError("wrong held recognition manifest")
    held, _ = load_bank(pins_root, diet_dir, data, "held", arms=("c",))
    pins = {(f"{case.cohort}_{case.person}", f"{case.order}_{case.visit}"): case for case in held}
    result = dict(control_sha256=sprint.CANDIDATE_SHA, phases={})
    total_activity = dict.fromkeys(sprint.ARMS, 0)
    for phase in MODES:
        result["phases"][phase] = {}
        orders = ORDERS[phase]
        for scheme in control.SCHEMES:
            files, active = set(), dict.fromkeys(sprint.ARMS, 0)

            def load(fold, order, key):
                shared = fold.startswith("new_") and (phase == "fresh" or scheme == "full")
                if shared and phase == "held":
                    case = pins[fold, key]
                    assert (case.scores["rec"] == -1).all()
                    np.testing.assert_array_equal(case.scores["c"], case.scores["half"])
                    return case.rows, {**case.scores, **{a: case.scores["c"] for a in sprint.ARMS}}
                path = (archive_path(history, scheme, fold, order) if shared
                        else args.out / phase / scheme / f"{fold}-{order}.npz")
                tape = read(path)
                files.add(path)
                if json.loads(str(tape["receipt"])) != receipts[fold]:
                    raise ValueError(f"wrong fold receipt in {path}")
                rows, ref = (tape[f"{field}_{key}"] for field in ("rows", "c"))
                age, identity = (int(tape[f"{field}_{key}"]) for field in ("rec", "identity"))
                if not (age == -1 and identity == 0 or 1 <= age <= len(rows) and 1 <= identity <= 10):
                    raise ValueError(f"invalid recognition receipt in {path}/{key}")
                if shared:
                    assert age == -1
                    np.testing.assert_array_equal(ref, tape[f"half_{key}"])
                labels = {arm: ref if shared else tape[f"{arm}_{key}"] for arm in sprint.ARMS}
                for arm, y in labels.items():
                    if y.shape != rows.shape or y.dtype != np.int8 or not np.isin(y, (0, 1, 2)).all():
                        raise ValueError(f"invalid {arm} labels in {path}/{key}")
                    prefix = age - 1 if age > 0 else len(rows)
                    np.testing.assert_array_equal(y[:prefix], ref[:prefix])
                    if not shared:
                        active[arm] += float(tape[f"activity_{arm}_{key}"]) > 2e-5
                if phase == "held" and scheme == "full":
                    case = pins[fold, key]
                    np.testing.assert_array_equal(rows, case.rows)
                    np.testing.assert_array_equal(ref, case.scores["c"])
                    assert age == int(case.scores["rec"][0]) and identity == int(case.scores["identity"][0])
                return rows, dict(half=ref, **labels, rec=np.full(len(rows), age), identity=np.full(len(rows), identity))

            deliveries = {seed: control.protocol.plans(data, seed, streams, scheme=scheme) for seed in orders}
            bank = control.read_bank(data, orders, load, deliveries=deliveries)
            for arm in sprint.ARMS:
                total_activity[arm] += active[arm]
            summary = control.report(bank, orders, arms=sprint.ARMS, nested_orders=True)
            for arm in sprint.ARMS:
                row = summary["arms"][arm]
                del row["passes_rule"]
                point, low, _ = row["delta"]["J"]
                row["passes"] = bool(point >= 0.05 and low >= 0 and row["changed_labels"]["new"] == 0)
            result["phases"][phase][scheme] = dict(**summary, active_streams=active, orders=orders,
                captures_sha256={str(p): control.digest(p) for p in sorted(files)})
    if not all(total_activity.values()):
        raise ValueError(f"an intervention never changed its route scores: {total_activity}")
    control.freeze(args.out / "proteus.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "report", *MODES))
    for arg in ("data", "fixtures", "history", "out"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--task", type=int)
    args = parser.parse_args()
    if args.mode == "report":
        collect(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
