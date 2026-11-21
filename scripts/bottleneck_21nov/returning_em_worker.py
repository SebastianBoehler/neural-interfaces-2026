"""Replay a recognition-gated SessionEM likelihood with the full causal tail."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch as monkeypatch

import numpy as np
import torch

from capture import DEVICE, DATA_SHA, V, fixture_assets, open_fold, pin, top_gate
from replay import taped
from tape import replay, tap
from returning_em import ARMS, patch, returning_em_output
from grow_replay import check, load
from dose import sessions

ORDERS = {"fit": tuple(range(71800, 71808)), "held": tuple(range(71808, 71816))}


def modified(template, items, arm, *, route=False):
    em = type(template.routeB.tail.em)
    original = em.step
    age = 0

    def step(owner, x, p):
        nonlocal age
        q = original(owner, x, p)
        weight = ARMS[arm] if items[age].recognised else 1.0
        age += 1
        return returning_em_output(p, q, weight, owner.state is not None)

    if template.v6r:
        raise ValueError("the scored upstream fine-tune path must remain independent")
    with monkeypatch.object(em, "step", step):
        result = replay(template, items, arm, trace_route=route)
    if age != len(items):
        raise ValueError("SessionEM did not consume every delivery")
    return result


def fresh(template):
    return V.D.official()[1].fresh_model(template)


def smoke(template, package, data, signals, rows, items, receipt, roster):
    reference = np.array([item.label for item in items])
    flags = np.array([item.recognised for item in items])
    prefix = rows[:84]
    original_text = (package / "submission.py").read_text()
    with TemporaryDirectory() as temp:
        folder = Path(temp) / roster
        folder.mkdir()
        for source in package.iterdir():
            if source.name != "submission.py":
                (folder / source.name).symlink_to(source.resolve())
        for arm in ARMS:
            (folder / "submission.py").write_text(patch(original_text, ARMS[arm]))
            live, _ = load(data, folder.parent, roster, receipt)
            seen = tap(live, signals, prefix, fresh, check, DEVICE)
            labs, scores, routes = modified(template, items, arm, route=True)
            np.testing.assert_array_equal(labs[~flags], reference[~flags])
            np.testing.assert_array_equal(labs[:len(prefix)], [d.label for d in seen], err_msg=arm)
            np.testing.assert_allclose(scores[:len(prefix)], [d.terminal for d in seen], rtol=0, atol=2e-5, err_msg=arm)
            np.testing.assert_allclose(routes[:len(prefix)], [d.route_out for d in seen], rtol=0, atol=2e-5, err_msg=arm)
            if arm != "native" and flags[:len(prefix)].any():
                if np.max(np.abs(routes[:len(prefix)] - [d.route_out for d in items[:len(prefix)]])) <= 1e-5:
                    raise ValueError(f"{arm}: inactive recognised SessionEM path")
            del live
    print(f"PASS {roster}: 84-window exact live labels, route/terminal score parity, active arms", flush=True)


def run(args):
    if not torch.cuda.is_available() or V.digest(args.data) != DATA_SHA:
        raise ValueError("CUDA and signed PROTEUS export required")
    roster, fold = args.roster, args.fold
    data = V.D.load(args.data)
    signals = torch.from_numpy(data["X"])
    pkg, receipts = open_fold(args.fixture, fold, "ba-confirm")
    receipt = receipts[roster]
    people = fixture_assets(pkg / roster, receipt)
    model, _ = load(data, pkg, roster, receipt)
    streams = sessions(data, fold)
    orders = ORDERS[args.phase][args.chunk * 4:args.chunk * 4 + 4] if not args.smoke else (71800,)
    if not orders:
        raise ValueError("empty order chunk")
    if args.smoke:
        streams = streams[:1]
    payload = {"receipt": np.asarray(json.dumps(receipt, sort_keys=True))}
    zero_total = 0
    for order in orders:
        for stream, rows in V.D.plans(data, order, streams).items():
            items = taped(model, signals, rows)
            base = np.array([item.label for item in items], dtype=np.int8)
            pin(args.fixture, fold, roster, order, stream, rows, base, receipt)
            replayed, _ = replay(model, items, "w25")
            np.testing.assert_array_equal(replayed, base)
            flags = np.array([item.recognised for item in items])
            age, identified = -1, ""
            if flags.any():
                age = int(flags.argmax()) + 1
                gate = top_gate(items[age - 1].contrast, items[age - 1].identity)
                if not gate or gate[2] <= 0 or not flags[age - 1:].all():
                    raise ValueError("recognition did not latch")
                identified = people[gate[0]]
            elif args.smoke and fold.startswith("ret_"):
                raise ValueError("returning smoke did not trigger recognition")
            if args.smoke:
                smoke(model, pkg / roster, data, signals, rows, items, receipt, roster)
            key = f"{order}_{stream.split('|')[1]}"
            payload.update({f"rows_{key}": rows, f"native_{key}": base,
                            f"rec_{key}": np.asarray(age), f"identity_{key}": np.asarray(identified)})
            for arm in ("half", "bypass"):
                y = modified(model, items, arm)[0] if flags.any() else base
                np.testing.assert_array_equal(y[~flags], base[~flags], err_msg=f"{fold}/{key}/{arm}")
                payload[f"{arm}_{key}"] = y
            zero_total += np.count_nonzero(~flags)
            print(f"PASS {fold} {roster} {key}: {len(rows)} pinned labels, recognition {age} {identified}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out) as temp:
        basename = f"{roster}-{fold}-{orders[0]}.npz"
        partial = Path(temp) / basename
        np.savez_compressed(partial, **payload)
        partial.rename(args.out / basename)
    print(f"PASS {zero_total} unrecognised deliveries unchanged", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data", type=Path)
    p.add_argument("fixture", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("roster", choices=("nine25", "ba_six"))
    p.add_argument("fold")
    p.add_argument("phase", choices=ORDERS)
    p.add_argument("--chunk", type=int, choices=(0, 1), default=0)
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
