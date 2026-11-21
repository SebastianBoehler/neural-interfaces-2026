"""Full Solver streams on paired selection or held-out confirmation folds."""
import argparse
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from folds import DATA_SHA, D, ORDERS, pinned_pair, streams
from hedge import CONFIRM, LEAD_ARM, SCHEDULES, SELECT, revised
from live import digest, fold_model, stream


def check_calls(model, rows, schedule, zero):
    expected = len(rows) if schedule is not None and not zero else 0
    assert model.hedge_seen == expected and 0 <= model.hedge_agreed <= expected
    if not zero:
        assert len(model.team_b.windows) == len(rows)
        graphs = model.team_b.feature_graphs
        assert not any(g.failed for g in graphs.values())
        assert graphs[1].replays > 0 and (len(rows) < 64 or graphs[64].replays > 0)
    assert model.routeB.log["err"] == 0


def run(template, signals, rows, schedule, zero=False, capture=False):
    assert template.hedge_seen == template.hedge_agreed == 0 and not template.team_b.windows
    template.ROUND2_HEDGE = schedule
    posterior = []

    def observe(model):
        posterior.append(model.trace["v6"]["mix6"].copy())

    def finish(model):
        check_calls(model, rows, schedule, zero)

    labels = stream(template, signals, rows, zero=zero, inspect=observe if capture else None, finish=finish)
    assert template.hedge_seen == template.hedge_agreed == 0 and not template.team_b.windows
    return labels, np.asarray(posterior)


def check_formula(model):
    pa = np.array([0.7, 0.2, 0.1])
    pb = np.array([0.2, 0.7, 0.1])
    clone = D.official()[1].fresh_model(model)
    for arm, schedule in SCHEDULES.items():
        if schedule is None:
            continue
        clone.reset_state()
        clone.ROUND2_HEDGE = schedule
        w0, late, tau, disagreement = schedule
        for t, q in enumerate((pb, pa, pb), 1):
            agreement = (0, 0.5, 1 / 3)[t - 1]
            w = np.clip(late + (w0 - late) * np.exp(-t / tau) + disagreement * (1 - agreement), 0, 1)
            expected = pa ** (1 - w) * q ** w
            np.testing.assert_allclose(clone.hedge_pool(pa, q), expected / expected.sum(), rtol=1e-14, atol=0)
        assert (clone.hedge_seen, clone.hedge_agreed) == (3, 1)
    clone.reset_state()
    assert (clone.hedge_seen, clone.hedge_agreed) == (0, 0)


def smoke(data, signals, model, fold):
    check_formula(model)
    seed = ORDERS[0]
    plans = D.plans(data, seed, streams(data, fold))
    for stream_key, rows in list(plans.items())[:1]:
        base, lead, _ = pinned_pair(fold, seed, stream_key, rows)
        np.testing.assert_array_equal(run(model, signals, rows, SCHEDULES["default"], zero=True)[0], base)
        control, mixer = run(model, signals, rows, None, capture=True)
        np.testing.assert_array_equal(control, lead)
        for arm in ("default", "late_ramp"):
            prefix = rows[:72]
            labels, pool = run(model, signals, prefix, SCHEDULES[arm], capture=True)
            labels_again, pool_again = run(model, signals, prefix[:16], SCHEDULES[arm], capture=True)
            np.testing.assert_array_equal(labels_again, labels[:16])
            np.testing.assert_allclose(pool_again, pool[:16], atol=2e-6, rtol=0)
            assert np.max(np.abs(pool - mixer[:72])) > 1e-3
        print(f"PASS {fold} 190-window-class pinned parity on {len(rows)} rows, live P3 and prefix reset", flush=True)


def evaluate(data, signals, model, fold, start, receipt, result, frozen):
    orders = (CONFIRM if frozen is not None else SELECT)[start:start + 2]
    arms = list(SCHEDULES) if frozen is None else [LEAD_ARM, frozen["chosen"]]
    captured = dict(orders=np.asarray(orders), arms=np.asarray(arms), **receipt)
    for order in orders:
        for stream_key, rows in D.plans(data, order, streams(data, fold)).items():
            key = f"{order}_{stream_key.split('|')[1]}"
            captured[f"rows_{key}"] = rows
            pivot = (order - orders[0]) % len(arms)
            for arm in arms[pivot:] + arms[:pivot]:
                t = time.perf_counter()
                captured[f"{arm}_{key}"] = run(model, signals, rows, SCHEDULES[arm])[0]
                print(f"{fold} {key} {arm} {len(rows)} windows {time.perf_counter() - t:.2f} s", flush=True)
    target = result / f"{fold}_{orders[0]}.npz"
    with target.open("xb") as output:
        np.savez_compressed(output, **captured)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("smoke", "select", "confirm"))
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--fold", required=True)
    p.add_argument("--chunk", type=int, choices=(0, 1, 2, 3), default=0)
    args = p.parse_args()
    assert torch.cuda.is_available() and "A100" in torch.cuda.get_device_name()
    assert digest(args.data) == DATA_SHA
    data = D.load(args.data)
    signals = torch.from_numpy(data["X"])
    with TemporaryDirectory() as directory:
        model, weights, source = fold_model(data, args.fold, Path(directory), source_patch=revised)
        if args.phase == "smoke":
            smoke(data, signals, model, args.fold)
        else:
            decision = args.run / "select.json"
            frozen = json.loads(decision.read_text()) if args.phase == "confirm" else None
            receipt = dict(source_sha=source, weights_sha=weights, data_sha=DATA_SHA)
            if frozen:
                receipt["selection_sha"] = digest(decision)
            evaluate(data, signals, model, args.fold, 2 * args.chunk, receipt, args.run / args.phase, frozen)


if __name__ == "__main__":
    main()
