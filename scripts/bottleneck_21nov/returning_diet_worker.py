"""Rerun the Last Push tail only on canonically recognised ba_six streams."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch as monkeypatch

import numpy as np
import torch

from returning_diet import ARMS, diet_tail_step, patch
from returning_em import patch as patch_em
from returning_em_worker import DATA_SHA, DEVICE, V, check, fixture_assets, fresh, load, modified, open_fold, pin, taped
from tape import replay, tap

ROSTER = "ba_six"


def diet(model, items, arm):
    tail_cls = type(model.routeB.tail)
    stock_step = tail_cls.step

    def before_route(route, item):
        route.tail.diet_weight = ARMS[arm] if item.recognised else None
        return not item.recognised

    def step(tail, x, u, mix):
        if tail.diet_weight is None:
            return stock_step(tail, x, u, mix)
        return diet_tail_step(tail, x, u, mix)

    with monkeypatch.object(tail_cls, "step", step):
        return replay(model, items, arm, trace_route=True, before_route=before_route)


def stage(folder, original, source):
    folder.mkdir(parents=True)
    for f in original.iterdir():
        if f.name != "submission.py":
            (folder / f.name).symlink_to(f.resolve())
    (folder / "submission.py").write_text(source)


def smoke(model, folder, data, signals, rows, items, receipt, age):
    prefix = rows[:84]
    half_source = patch_em((folder / "submission.py").read_text(), 0.5)
    with TemporaryDirectory() as tmp:
        for arm in ("half", *ARMS):
            source = half_source if arm == "half" else patch(half_source, ARMS[arm])
            root = Path(tmp) / arm
            stage(root / ROSTER, folder, source)
            live, _ = load(data, root, ROSTER, receipt)

            def check_live(m, n):
                check(m, n)
                expected_count = n if arm == "half" else age - 1
                expected_em = age - 1 if arm == "zero" else n
                tail = m.routeB.tail
                if (m.log6["rec"] != age - 1 or m.cf.t != expected_count or m.cp.t != n or
                        len(tail.em.A) != expected_em or len(tail.tsem.U) != expected_em or
                        len(tail.rb.hist) != min(tail.rb.W, n)):
                    raise ValueError(f"{arm}: live latch or adaptation counts differ")
                if arm == "quarter" and (tail.em.state is None or tail.tsem.state is None):
                    raise ValueError("the quarter arm did not fit both EM stages")

            seen = tap(live, signals, prefix, fresh, check_live, DEVICE)
            result = modified(model, items, "half", route=True) if arm == "half" else diet(model, items, arm)
            for name, scores in zip(("label", "terminal", "route_out"), result, strict=True):
                recorded = np.asarray([getattr(d, name) for d in seen])
                if name == "label":
                    np.testing.assert_array_equal(scores[:len(prefix)], recorded, err_msg=arm)
                else:
                    np.testing.assert_allclose(scores[:len(prefix)], recorded, rtol=0, atol=2e-5, err_msg=arm)
            if arm != "half":
                cut = diet(model, items[:len(prefix)], arm)
                for short, full in zip(cut, result, strict=True):
                    np.testing.assert_array_equal(short, full[:len(prefix)])
            del live
    print("PASS exact live labels, route/terminal score tolerance, stage activity and exact tape prefixes", flush=True)


def run(a):
    if not torch.cuda.is_available() or V.digest(a.data) != DATA_SHA:
        raise ValueError("CUDA and signed PROTEUS data required")
    blocks = json.loads(a.manifest.read_text())["active"]
    block, = [b for b in blocks if b["roster"] == ROSTER and b["task"] == a.task]
    fold, phase = block["fold"], block["phase"]
    active = block["recognition"]
    first = min(int(key.split("_")[0]) for key in active)
    first -= first % 4
    if a.smoke:
        key = min(active)
        active = {key: active[key]}
    data = V.D.load(a.data)
    signals = torch.from_numpy(data["X"])
    pkg, receipts = open_fold(a.fixtures, fold, "ba-confirm")
    receipt = receipts[ROSTER]
    fixture_assets(pkg / ROSTER, receipt)
    model, _ = load(data, pkg, ROSTER, receipt)
    payload = {"receipt": np.asarray(json.dumps(receipt, sort_keys=True))}
    archive = a.archive / phase / f"{ROSTER}-{fold}-{first}.npz"
    with np.load(archive, allow_pickle=False) as expected:
        if str(expected["receipt"]) != str(payload["receipt"]):
            raise ValueError("em_half archive used different fold assets")
        for key, age in sorted(active.items()):
            order, visit = key.split("_")
            stream = f"{fold.split('_')[1]}|{visit}"
            rows = V.D.plans(data, int(order), [stream])[stream]
            items = taped(model, signals, rows)
            native = np.array([item.label for item in items])
            pin(a.fixtures, fold, ROSTER, int(order), stream, rows, native, receipt)
            np.testing.assert_array_equal(replay(model, items, "w25")[0], native)
            flags = np.arange(len(rows)) >= age - 1
            np.testing.assert_array_equal(flags, [item.recognised for item in items])
            np.testing.assert_array_equal(rows, expected[f"rows_{key}"])
            np.testing.assert_array_equal(native, expected[f"native_{key}"])
            if age != int(expected[f"rec_{key}"]):
                raise ValueError("em_half baseline recognition age mismatch")
            half, terminal, route = modified(model, items, "half", route=True)
            np.testing.assert_array_equal(half, expected[f"half_{key}"])
            payload[f"rows_{key}"] = rows
            payload[f"half_{key}"] = half
            for arm in ARMS:
                y, q, route_out = diet(model, items, arm)
                np.testing.assert_array_equal(q[~flags], terminal[~flags])
                np.testing.assert_array_equal(route_out[~flags], route[~flags])
                np.testing.assert_array_equal(y[~flags], half[~flags])
                if a.smoke and np.max(np.abs(route_out[flags] - route[flags])) <= 2e-5:
                    raise ValueError(f"{arm}: posterior unchanged after latch")
                payload[f"{arm}_{key}"] = y
            if a.smoke:
                smoke(model, pkg / ROSTER, data, signals, rows, items, receipt, age)
            print(f"PASS {fold} {key}: {len(rows)} pinned native/em_half labels, latch {age}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=a.out) as temp:
        target = Path(temp) / f"{fold}-{first}.npz"
        np.savez_compressed(target, **payload)
        target.rename(a.out / target.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "fixtures", "archive", "manifest", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("task", type=int)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
