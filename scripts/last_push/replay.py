"""Run the paired weight and first-40 replay on signed fused fold fixtures."""
import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import torch

from patch import DOSES, EARLY, PARENTS, ROOT, patch
from tape import replay, tap

sys.path[:0] = [str(ROOT / f"scripts/{d}") for d in ("leonardo_fill", "laneTB")]
from grow_replay import check, load  # noqa: E402
from grow_fixture import open_fold  # noqa: E402
from dose import DATA_SHA, V, sessions  # noqa: E402

ORDERS = {"fit": tuple(range(71800, 71808)), "held": tuple(range(71808, 71816))}
DEVICE = V.DEVICE


def taped(model, signals, rows):
    return tap(model, signals, rows, V.D.official()[1].fresh_model, check, DEVICE)


def pin(path, fold, roster, order, stream, rows, labels, receipt):
    first = order - order % 2
    path = path / "held" / fold / roster / f"pred_{fold}_{first}.npz"
    key = f"{order}_{stream.split('|')[1]}"
    with np.load(path, allow_pickle=False) as tape:
        if json.loads(str(tape["receipt"])) != receipt:
            raise ValueError(f"fold receipt differs from pin: {path}")
        np.testing.assert_array_equal(rows, tape[f"rows_{key}"])
        np.testing.assert_array_equal(labels, tape[f"labels_{key}"])


def smoke(model, pkg, data, signals, rows, delivery):
    expected = np.array([d.label for d in delivery], dtype=np.int8)
    labels, stock_scores = replay(model, delivery, "w25")
    np.testing.assert_array_equal(labels, expected)
    prefix = rows[:72]
    for arm in ("w40", *EARLY):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for file in pkg.iterdir():
                if file.name != "submission.py":
                    (folder / file.name).symlink_to(file.resolve())
            file = folder / "submission.py"
            file.write_text(patch((pkg / file.name).read_text(), arm))
            live, _ = V.timed_load(folder, data, DEVICE)
            labels, captures, _ = V.stream_session(live, signals, prefix, DEVICE, capture=True, check=check)
            expected, scores, routes = replay(model, delivery, arm, trace_route=True)
            cut_labels, cut_scores = replay(model, delivery[:len(prefix)], arm)
            np.testing.assert_array_equal(labels, expected[:len(prefix)])
            np.testing.assert_array_equal(cut_labels, expected[:len(prefix)])
            np.testing.assert_array_equal(cut_scores, scores[:len(prefix)])
            if arm in DOSES:
                mixed = np.stack([(1 - DOSES[arm]) * d.base + DOSES[arm] * d.team_b
                                  for d in delivery[:len(prefix)]])
                np.testing.assert_allclose(captures[:, :3], mixed, rtol=0, atol=2e-5)
            np.testing.assert_allclose(captures[:, 3:], routes[:len(prefix)], rtol=0, atol=2e-5)
            if np.max(np.abs(scores - stock_scores)) <= 1e-7:
                raise ValueError(f"{arm} had no effect during smoke")
            print(f"PASS {arm}: live {len(prefix)} labels and route scores, exact tape prefix", flush=True)


def production_smoke(data, signals, args):
    if V.digest(args.zip) != PARENTS[args.roster]:
        raise ValueError("wrong signed production ZIP")
    with TemporaryDirectory() as tmp:
        pkg = Path(tmp)
        with ZipFile(args.zip) as z:
            z.extractall(pkg)
        if patch((pkg / "submission.py").read_text(), "w25") != (pkg / "submission.py").read_text():
            raise ValueError("25% source is not identical")
        model, _ = V.timed_load(pkg, data, DEVICE)
        if model.TEAM_B_MIX != 0.25 or len(model.team_b.nets) != (6 if args.roster == "ba_six" else 9):
            raise ValueError("wrong loaded production roster")
        rows = V.D.plans(data, 71800, ["P01|1"])["P01|1"]
        smoke(model, pkg, data, signals, rows, taped(model, signals, rows))
    print(f"PASS {args.roster} production smoke", flush=True)


def fold_replay(data, signals, args):
    pkg, receipts = open_fold(args.fixture, args.fold, "ba-confirm")
    receipt = receipts[args.roster]
    model, _ = load(data, pkg, args.roster, receipt)
    if args.phase == "smoke":
        stream = sessions(data, args.fold)[0]
        rows = V.D.plans(data, 71800, [stream])[stream]
        delivery = taped(model, signals, rows)
        pin(args.fixture, args.fold, args.roster, 71800, stream, rows,
            np.array([d.label for d in delivery]), receipt)
        smoke(model, pkg / args.roster, data, signals, rows, delivery)
        print(f"PASS {args.fold} {args.roster}: {len(rows)} pinned labels", flush=True)
        return
    orders = ORDERS[args.phase][2 * args.chunk:2 * args.chunk + 2]
    if len(orders) != 2:
        raise ValueError("invalid order chunk")
    arms = tuple(DOSES)
    if args.phase == "held":
        selection = json.loads((args.run / "fit.json").read_text())
        if selection["orders"] != list(ORDERS["fit"]):
            raise ValueError("selection used the wrong orders")
        arms = tuple(dict.fromkeys(("w25", selection["rosters"][args.roster]["weight"])))
    arms += tuple(EARLY)
    payload = {"receipt": np.asarray(json.dumps(receipt, sort_keys=True)), "arms": np.asarray(arms)}
    for order in orders:
        for stream, rows in V.D.plans(data, order, sessions(data, args.fold)).items():
            key = f"{order}_{stream.split('|')[1]}"
            delivery = taped(model, signals, rows)
            labels = np.array([d.label for d in delivery], dtype=np.int8)
            pin(args.fixture, args.fold, args.roster, order, stream, rows, labels, receipt)
            payload[f"rows_{key}"] = rows
            for arm in arms:
                y, _ = replay(model, delivery, arm)
                if arm == "w25":
                    np.testing.assert_array_equal(y, labels)
                payload[f"{arm}_{key}"] = y
            print(f"PASS {args.fold} {args.roster} {key} {len(rows)}", flush=True)
    folder = args.run / args.phase / args.roster / args.fold
    folder.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=folder) as tmp:
        target = Path(tmp) / f"{orders[0]}.npz"
        np.savez_compressed(target, **payload)
        target.rename(folder / target.name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("production-smoke", "smoke", *ORDERS))
    p.add_argument("data", type=Path)
    p.add_argument("roster", choices=tuple(PARENTS))
    p.add_argument("--zip", type=Path)
    p.add_argument("--fixture", type=Path)
    p.add_argument("--run", type=Path)
    p.add_argument("--fold")
    p.add_argument("--chunk", type=int)
    args = p.parse_args()
    if not torch.cuda.is_available() or V.digest(args.data) != DATA_SHA:
        raise ValueError("CUDA and signed PROTEUS export required")
    data = V.D.load(args.data)
    signals = torch.from_numpy(data["X"])
    if args.phase == "production-smoke":
        production_smoke(data, signals, args)
    else:
        fold_replay(data, signals, args)


if __name__ == "__main__":
    main()
