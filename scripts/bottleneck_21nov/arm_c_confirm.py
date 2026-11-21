"""Confirm the frozen SessionEM-only arm on fresh PROTEUS delivery orders."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from capture import top_gate
from diet_decomposition_patch import ARMS, patch
from diet_decomposition_tail import intervene
from oracles import freeze
from quarter_protocol import FOLDS, ROSTER
from returning_diet_report import PARENT_SHA, digest, report
from returning_diet_worker import DATA_SHA, V, fixture_assets, load, modified, open_fold, replay, stage, taped
from returning_em import patch as patch_half
from returning_em_worker import DEVICE, check, fresh, sessions, tap
from tapes import metadata, protocol, read_bank

ORDERS = (72120, 72121, 72122, 72123)
SCHEMES = ("full", "within")
CANDIDATE_SHA = "cba9b208fbff7ce416efd8b5ce68ae1ad0771980d84e754433bd33e9996d0bca"


def check_prefix(model, package, data, signals, rows, items, receipt, age):
    prefix = rows[:84]
    half = modified(model, items, "half", route=True)
    for existing, repeated in zip(half, intervene(model, items, "half"), strict=True):
        np.testing.assert_array_equal(existing, repeated)
    candidate = intervene(model, items, "c")
    if age < 1 or age > len(prefix) or np.max(np.abs(candidate[2][:len(prefix)] - half[2][:len(prefix)])) <= 2e-5:
        raise ValueError("returning prefix must exercise the SessionEM weight")
    for full, short in zip(candidate, intervene(model, items[:len(prefix)], "c"), strict=True):
        np.testing.assert_array_equal(full[:len(prefix)], short)

    original = (package / "submission.py").read_text()
    with TemporaryDirectory() as temp:
        folder = Path(temp)
        stage(folder / ROSTER, package, patch(patch_half(original, 0.5), ARMS["c"]))
        live, _ = load(data, folder, ROSTER, receipt)
        seen = tap(live, signals, prefix, fresh, check, DEVICE)
    np.testing.assert_array_equal([item.recognised for item in seen], np.arange(len(prefix)) >= age - 1)
    np.testing.assert_array_equal([item.label for item in seen], candidate[0][:len(prefix)])
    for key, expected in zip(("terminal", "route_out"), candidate[1:], strict=True):
        np.testing.assert_allclose([getattr(item, key) for item in seen], expected[:len(prefix)], atol=2e-5, rtol=0)


def score(data_path, fixtures, out, task, *, smoke=False):
    if not torch.cuda.is_available() or V.digest(data_path) != DATA_SHA:
        raise ValueError("CUDA and signed PROTEUS data required")
    order_index, fold_index = divmod(task, len(FOLDS))
    order, fold = ORDERS[order_index], FOLDS[fold_index]
    data = V.D.load(data_path)
    signals = torch.from_numpy(data["X"])
    package, receipts = open_fold(fixtures, fold, "ba-confirm")
    receipt = receipts[ROSTER]
    people = fixture_assets(package / ROSTER, receipt)
    model, _ = load(data, package, ROSTER, receipt)
    streams = sessions(data, fold)
    if smoke:
        streams = streams[-1:]
    for scheme in SCHEMES:
        payload = {"receipt": np.asarray(json.dumps(receipt, sort_keys=True))}
        for stream, rows in V.D.plans(data, order, streams, scheme=scheme).items():
            items = taped(model, signals, rows)
            native = np.asarray([item.label for item in items], dtype=np.int8)
            np.testing.assert_array_equal(replay(model, items, "w25")[0], native)
            flags = np.asarray([item.recognised for item in items])
            age, identity = -1, 0
            half = candidate = native
            if flags.any():
                i = int(flags.argmax())
                gate = top_gate(items[i].contrast, items[i].identity)
                if gate is None or gate[2] <= 0 or not flags[i:].all():
                    raise ValueError(f"invalid recognition latch: {fold}/{order}/{stream}")
                age, identity = i + 1, int(people[gate[0]][1:])
                half, half_scores, half_route = modified(model, items, "half", route=True)
                candidate, scores, route = intervene(model, items, "c")
                np.testing.assert_array_equal(half[~flags], native[~flags])
                np.testing.assert_array_equal(candidate[~flags], half[~flags])
                np.testing.assert_array_equal(scores[~flags], half_scores[~flags])
                np.testing.assert_array_equal(route[~flags], half_route[~flags])
            key = f"{order}_{stream.split('|')[1]}"
            payload.update({f"rows_{key}": rows, f"half_{key}": half, f"c_{key}": candidate,
                            f"rec_{key}": np.asarray(age), f"identity_{key}": np.asarray(identity)})
            print(f"PASS {scheme} {fold}/{key}: {len(rows)} rows, recognition {age}, identity {identity}", flush=True)
            if smoke:
                check_prefix(model, package / ROSTER, data, signals, rows, items, receipt, age)
        if not smoke:
            folder = out / scheme
            with TemporaryDirectory(dir=folder) as tmp:
                path = Path(tmp) / f"{fold}-{order}.npz"
                np.savez_compressed(path, **payload)
                path.rename(folder / path.name)


def collect(data_path, fixtures, captures, output):
    data = metadata(data_path)
    with np.load(data_path, allow_pickle=False) as source:
        data.update({key: source[key] for key in ("run", "onset")})
    streams = np.unique(data["stream"])
    receipts = {fold: json.loads((fixtures / "fixtures" / fold / "receipt.json").read_text())[ROSTER]
                for fold in FOLDS}
    result = dict(orders=ORDERS, parent_sha256=PARENT_SHA, candidate_sha256=CANDIDATE_SHA, schemes={})
    for scheme in SCHEMES:
        files = {}

        def read(fold, order, key):
            path = captures / scheme / f"{fold}-{order}.npz"
            if path not in files:
                with np.load(path, allow_pickle=False) as tape:
                    files[path] = dict(tape)
                if json.loads(str(files[path]["receipt"])) != receipts[fold]:
                    raise ValueError(f"wrong fold receipt in {path}")
            tape = files[path]
            rows, ref, candidate = (tape[f"{field}_{key}"] for field in ("rows", "half", "c"))
            age, identity = (int(tape[f"{field}_{key}"]) for field in ("rec", "identity"))
            if any(labels.shape != rows.shape or labels.dtype != np.int8 or not np.isin(labels, (0, 1, 2)).all()
                   for labels in (ref, candidate)):
                raise ValueError(f"invalid labels in {path}/{key}")
            if not (age == -1 and identity == 0 or 1 <= age <= len(rows) and 1 <= identity <= 10):
                raise ValueError(f"invalid recognition receipt in {path}/{key}")
            prefix = age - 1 if age > 0 else len(rows)
            np.testing.assert_array_equal(candidate[:prefix], ref[:prefix])
            return rows, dict(half=ref, c=candidate, rec=np.full(len(rows), age), identity=np.full(len(rows), identity))

        deliveries = {seed: protocol.plans(data, seed, streams, scheme=scheme) for seed in ORDERS}
        bank = read_bank(data, ORDERS, read, deliveries=deliveries)
        summary = report(bank, ORDERS, arms=("c",), nested_orders=True)
        del summary["arms"]["c"]["passes_rule"]  # The original threshold applies to the joint quarter diet.
        result["schemes"][scheme] = dict(**summary, captures_sha256={p.name: digest(p) for p in sorted(files)})
    freeze(output, result)
    print(f"Collected both schemes into {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "score", "report"))
    for arg in ("data", "fixtures", "out"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--task", type=int, choices=range(len(FOLDS) * len(ORDERS)))
    args = parser.parse_args()
    if args.mode == "report":
        collect(args.data, args.fixtures, args.out, args.out.parent / "report.json")
    else:
        if args.mode == "score" and args.task is None:
            parser.error("score requires --task")
        score(args.data, args.fixtures, args.out, 11 if args.mode == "smoke" else args.task,
              smoke=args.mode == "smoke")


if __name__ == "__main__":
    main()
