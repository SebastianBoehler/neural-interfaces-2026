"""Fit and replay the released Bet 3 on the registered folds."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from anchor_bank import aligners, attach, compute
from anchor_source import revised
from settings import DATA_SHA, DEVICE, D, FIT, FOLDS, FOLD_ASSETS, HELD, LEAD_SHA, ORDERS
from settings import digest, fold_model, streams
from variants import predict, replay
from worker import inspect_trace, pinned_fold, save

DOSES = (4, 8, 16)
RETURN_FOLDS = tuple(f"ret_{p}" for p in D.RETURNING)


def anchored(template, windows, n0, expected):
    prev = template.round3_n0
    template.round3_n0 = n0
    try:
        model = D.official()[1].fresh_model(template)
    finally:
        template.round3_n0 = prev
    if model.round3_started != -1 or model.team_b.round3_ref is not None:
        raise RuntimeError("enrolled reference crossed a session boundary")
    labels = np.fromiter((predict(model, x[None]) for x in windows), dtype=np.int8, count=len(windows))
    if model.log6["v6"] != len(windows):
        raise ValueError("degraded anchored stream")
    start = model.round3_started
    prefix = len(windows) if start < 0 else start
    np.testing.assert_array_equal(labels[:prefix], expected[:prefix])
    if start >= 0:
        assert n0 in DOSES and start == model.log6["rec"] + 1 and start < len(windows)
        refs = [p.round3_ref for p in (model.core1, model.core2, *aligners(model), model.team_b)]
        for (n, actual), target in zip(refs, model.round3_bank[model.round3_person], strict=True):
            assert n == n0
            if torch.is_tensor(actual):
                actual = actual.cpu().numpy()
            np.testing.assert_array_equal(actual, target)
    elif n0 and model.log6["rec"] >= 0:
        raise RuntimeError("recognized stream never received its anchor")
    return labels, start, model.round3_person


def smoke(data, model, fold, pinned):
    rows, (_, lead) = pinned
    windows = torch.from_numpy(data["X"][rows]).to(DEVICE)
    zero = replay(model, windows)["labels"]
    np.testing.assert_array_equal(zero, lead)
    if fold in RETURN_FOLDS:
        model.round3_bank = {}
        attach(model, dict(names=np.array(model.round3_names), **compute(model, data, fold)))
        for dose in DOSES:
            _, start, j = anchored(model, windows, dose, lead)
            assert 0 < start <= 30 and model.round3_names[j] == fold.split("_")[1]
        np.testing.assert_array_equal(anchored(model, windows, 0, lead)[0], lead)
        print(f"PASS {fold} zero and three active anchors at native recognition on {len(rows)} windows", flush=True)
    else:
        print(f"PASS {fold} source-patched native zero on {len(rows)} pinned windows", flush=True)


def run(args, data, model, identity, source):
    base = args.base
    path = args.run / "banks" / f"{args.fold}.npz"
    if args.stage == "bank":
        save(path, dict(names=np.array(model.round3_names), source_anchor_sha=source,
                        **identity, **compute(model, data, args.fold)))
        return
    with np.load(path, allow_pickle=False) as bank:
        for k, v in identity.items():
            assert str(bank[k]) == v
        assert str(bank["source_anchor_sha"]) == source
        attach(model, bank)
    if args.stage == "fit":
        assert args.fold in RETURN_FOLDS
        orders = FIT
        doses = DOSES
    else:
        orders = HELD
        selection = args.run / "select.json"
        report = json.loads(selection.read_text())
        doses = (report["dose"],)
        identity["selection_sha"] = digest(selection)
    chunk = orders[2 * args.chunk:2 * args.chunk + 2]
    filename = f"{args.fold}_{chunk[0]}.npz"
    result = dict(orders=np.array(chunk), doses=np.array(doses), bank_sha=digest(path),
                  source_anchor_sha=source, **identity)
    with np.load(base / "trace" / filename, allow_pickle=False) as trace:
        stable = {k: v for k, v in identity.items() if k != "selection_sha"}
        inspect_trace(trace, stable, chunk)
        for order in chunk:
            for stream, rows in D.plans(data, order, streams(data, args.fold)).items():
                key = f"{order}_{stream.split('|')[1]}"
                np.testing.assert_array_equal(rows, trace[f"rows_{key}"])
                ref = trace[f"lead_{key}"]
                windows = torch.from_numpy(data["X"][rows]).to(DEVICE)
                result[f"rows_{key}"] = rows
                for n0 in doses:
                    start = perf_counter()
                    labels, age, person = anchored(model, windows, n0, ref)
                    result.update({f"n{n0}_{key}": labels, f"age_n{n0}_{key}": age, f"person_n{n0}_{key}": person})
                    print(f"{args.stage} {args.fold} {key} n0={n0} age={age} gallery={person} {perf_counter()-start:.1f}s",
                          flush=True)
    save(args.run / args.stage / filename, result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("smoke", "bank", "fit", "held"))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--fold", choices=FOLDS, required=True)
    p.add_argument("--chunk", type=int, choices=range(4), default=0)
    args = p.parse_args()
    if not torch.cuda.is_available() or digest(args.data) != DATA_SHA:
        raise ValueError("expected the registered GPU and data")
    screen = json.loads((args.base / "screen.json").read_text())
    if not screen["step0_fit"]["bet3_released"]:
        raise ValueError("Step 0 did not release Bet 3")
    data = D.load(args.data)
    with TemporaryDirectory() as tmp:
        model, weights, source = fold_model(data, args.fold, Path(tmp), source_patch=revised)
        pinned = pinned_fold(data, args.fold, weights)
        if args.stage == "smoke":
            smoke(data, model, args.fold, pinned)
        else:
            base_weights = digest(FOLD_ASSETS / args.fold / "pkg" / "weights.npz")
            identity = dict(weights_sha=weights, base_weights_sha=base_weights, data_sha=DATA_SHA, lead_sha=LEAD_SHA,
                            source_sha=screen["folds"][args.fold]["source_sha"])
            assert screen["folds"][args.fold]["weights_sha"] == weights
            assert screen["folds"][args.fold]["base_weights_sha"] == base_weights
            run(args, data, model, identity, source)


if __name__ == "__main__":
    main()
