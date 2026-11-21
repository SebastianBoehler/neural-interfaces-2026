"""Generate blind fold tapes. This worker never computes accuracy."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import torch

from decisions import bet2_predict
from frozen import frozen_labels
from metrics import Session
from settings import DATA_SHA, D, DEVICE, FIT, FOLDS, FOLD_ASSETS, HELD, LEAD, LEAD_SHA, ORDERS, WEIGHTS
from settings import digest, fold_model, pinned_pair, streams, timed_load
from variants import RebalancedMix, replay


def public_model(data, work, archive):
    if digest(archive) != LEAD_SHA:
        raise ValueError("public smoke needs the signed nine25 lead ZIP")
    with ZipFile(archive) as z:
        z.extractall(work)
    return timed_load(work, data, DEVICE)[0]


def smoke(template, windows, data, rows, pinned=None):
    reference = replay(template, windows)
    with_taps, end = replay(template, windows, capture=True, endpoint=True)
    np.testing.assert_array_equal(reference["labels"], with_taps["labels"])
    raw = with_taps["team"]
    same = replay(template, windows, raw=raw, capture=True)
    np.testing.assert_array_equal(same["labels"], reference["labels"])
    np.testing.assert_allclose(same["pre_em"], with_taps["pre_em"], atol=2e-6, rtol=0)
    hard = replay(template, windows, hard=True, capture=True)
    sample = Session("new", "P01", "1", ORDERS[0], data["y"][rows], data["interface"][rows],
                     dict(pre_em=hard["pre_em"], hard=hard["labels"], hard_log=hard["terminal"], team=raw))
    np.testing.assert_array_equal(bet2_predict(sample, (0, 0)), hard["labels"])
    np.testing.assert_array_equal(bet2_predict(sample, (1, 1)), raw.argmax(1))
    if pinned:
        np.testing.assert_array_equal(reference["labels"], pinned[1])
        np.testing.assert_array_equal(hard["labels"], pinned[0])
        if not np.any(hard["labels"] != reference["labels"]):
            raise ValueError("pinned comparison did not exercise Team B")
    active = replay(template, windows, raw=raw, variant=RebalancedMix(0.25, 1), capture=True)
    assert np.max(np.abs(active["pre_em"] - with_taps["pre_em"])) > 1e-4
    np.testing.assert_allclose(replay(template, windows[:8], capture=True)["team"], raw[:8], atol=1e-7, rtol=0)
    frozen_labels(end, windows[:6], smoke=True)
    assert not template.team_b.windows and end.log6["v6"] == len(windows)
    print(f"PASS {len(windows)} windows: native/cache parity, pre-EM and terminal taps, both blend paths, "
          f"prefix and frozen query order, pinned labels checked={pinned is not None}", flush=True)


def pinned_fold(data, fold, weights):
    key = streams(data, fold)[0]
    order = ORDERS[0]
    rows = D.plans(data, order, [key])[key]
    base, lead, expected = pinned_pair(fold, order, key, rows)
    if expected != weights:
        raise ValueError(f"{fold} Team B SHA-256 differs from the pinned Lane TB bank")
    return rows, (base, lead)


def save(path, records):
    with path.with_suffix(".partial").open("xb") as output:
        np.savez_compressed(output, **records)
    path.with_suffix(".partial").rename(path)


def run_trace(model, windows):
    lead, end = replay(model, windows, capture=True, endpoint=True)
    frozen = frozen_labels(end, windows)
    hard = replay(model, windows, hard=True, capture=True)
    return dict(lead=lead["labels"], frozen=frozen, hard=hard["labels"], pre_em=hard["pre_em"],
                hard_log=hard["terminal"], team=lead["team"], team_rb=lead["team_rb"])


def run_variant(model, windows, result, key, arms, order, first):
    raw = result[f"team_{key}"]
    expected = result[f"lead_{key}"]
    control = replay(model, windows, raw=raw)["labels"]
    np.testing.assert_array_equal(control, expected)
    pivot = (order - first) % len(arms)
    return {name: replay(model, windows, raw=raw, variant=config)["labels"]
            for name, config in arms[pivot:] + arms[:pivot]}


def inspect_trace(tape, identity, orders):
    for k, value in identity.items():
        assert str(tape[k]) == value, k
    assert tuple(tape["orders"]) == tuple(orders)


def run(args, data, model, weights, source):
    orders = (FIT if args.block == "fit" else HELD)[2 * args.chunk:2 * args.chunk + 2]
    base_weights = digest(FOLD_ASSETS / args.fold / "pkg" / "weights.npz")
    identity = dict(source_sha=source, weights_sha=weights, base_weights_sha=base_weights,
                    data_sha=DATA_SHA, lead_sha=LEAD_SHA)
    records = dict(orders=np.asarray(orders), **identity)
    filenames = f"{args.fold}_{orders[0]}.npz"
    paths = None
    if args.stage != "trace":
        target = args.run / ("screen.json" if args.stage == "grid" else "select.json")
        decision = json.loads(target.read_text())
        paths = args.run / "trace" / filenames
        start = decision["bet1_start"]
        weights_to_run = WEIGHTS if args.stage == "grid" else (decision["bet1_weight"],)
        arms = [(f"b1_{w:.3f}", RebalancedMix(w, start)) for w in weights_to_run]
        if args.stage == "grid":
            arms.append(("geometric_25", RebalancedMix(0.25, 0)))
        records["decision_sha"] = digest(target)
    tape = None if paths is None else np.load(paths, allow_pickle=False)
    if tape is not None:
        inspect_trace(tape, identity, orders)
    try:
        for order in orders:
            for stream, rows in D.plans(data, order, streams(data, args.fold)).items():
                key = f"{order}_{stream.split('|')[1]}"
                started = perf_counter()
                windows = torch.from_numpy(data["X"][rows]).to(DEVICE)
                records[f"rows_{key}"] = rows
                if tape is None:
                    result = run_trace(model, windows)
                else:
                    np.testing.assert_array_equal(tape[f"rows_{key}"], rows)
                    result = run_variant(model, windows, tape, key, arms, order, orders[0])
                records.update({f"{name}_{key}": arr for name, arr in result.items()})
                print(args.stage, args.fold, key, len(rows), f"{perf_counter()-started:.1f}s", flush=True)
        save(args.run / args.stage / filenames, records)
    finally:
        if tape is not None:
            tape.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("public-smoke", "fold-smoke", "trace", "grid", "held"))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--run", type=Path)
    p.add_argument("--lead", type=Path, default=LEAD)
    p.add_argument("--fold", choices=FOLDS, default="new_P01")
    p.add_argument("--block", choices=("fit", "held"), default="fit")
    p.add_argument("--chunk", type=int, choices=range(4), default=0)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Galvani or Leonardo GPU worker required")
    if digest(args.data) != DATA_SHA:
        raise ValueError("unexpected PROTEUS export SHA-256")
    if args.stage == "grid" and args.block != "fit" or args.stage == "held" and args.block != "held":
        raise ValueError("stage does not allow that order block")
    data = D.load(args.data)
    with TemporaryDirectory() as folder:
        work = Path(folder)
        if args.stage == "public-smoke":
            model = public_model(data, work, args.lead)
            key = "P01|1"
            rows = D.plans(data, ORDERS[0], [key])[key][:72]
            smoke(model, torch.from_numpy(data["X"][rows]).to(DEVICE), data, rows)
            return
        model, weights, source = fold_model(data, args.fold, work)
        rows, pinned = pinned_fold(data, args.fold, weights)
        if args.stage == "fold-smoke":
            smoke(model, torch.from_numpy(data["X"][rows]).to(DEVICE), data, rows, pinned)
        else:
            run(args, data, model, weights, source)


if __name__ == "__main__":
    main()
