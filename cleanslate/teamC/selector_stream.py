"""Replay the C1 fit on the cap250 fold harness' named full-session orders."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from stream_eval import VARIANTS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import data as harness

ORDERS = (71000, 71001, 71002, 71003)
ARMS = ("A1", "A2")
DATA_SHA = "2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327"


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def read_input(path):
    if digest(path) != DATA_SHA:
        raise ValueError("unexpected PROTEUS export")
    with np.load(path, allow_pickle=False) as z:
        person = np.asarray([s.rsplit("/", 1)[-1] for s in z["subject"]])
        visits = z["session"]
        stream = np.char.add(np.char.add(person, "|"), visits)
        return {"X": z["X"], "stream": stream}


def check_stock_input(path, stock):
    with np.load(path, allow_pickle=False) as a, np.load(stock, allow_pickle=False) as b:
        for key in ("X", "subject", "session", "interface", "ch_names"):
            np.testing.assert_array_equal(a[key], b[key], err_msg=f"cap250 export differs at {key}")
    print(f"OK cap250 input {digest(stock)} is aligned with Team C", flush=True)


def expected_sources(mode, held, data):
    all_sessions = set(data["stream"])
    if mode == "new":
        return {s for s in all_sessions if s.split("|")[0] != held}
    excluded = {f"{p}|{v}" for p in harness.RETURNING for v in ("2", "3")}
    return all_sessions - excluded


def check_sources(pkg, mode, held, data):
    receipt = json.loads((pkg / "train.json").read_text())
    recipe = receipt["args"]
    if recipe["extra"] or recipe["epochs"] != 60 or recipe["rot"] != 0:
        raise ValueError(f"{pkg}: unexpected C1 recipe")
    if "streams" in receipt:
        sources = set(receipt["streams"])
    else:
        # The first C1 fits recorded people and row counts before exclusion was supported.
        if mode != "new" or "exclude" in recipe or sorted(recipe["people"].split(",")) != sorted(receipt["people"]):
            raise ValueError(f"{pkg}: incomplete training roster")
        sources = {s for s in data["stream"] if s.split("|")[0] in receipt["people"]}
    if sources != expected_sources(mode, held, data) or np.isin(data["stream"], list(sources)).sum() != receipt["rows"]:
        raise ValueError(f"{pkg}: incorrect training roster")
    return sources


def run(model, x, rows, streaming):
    torch.cuda.synchronize()
    start = time.perf_counter()
    session = streaming.fresh_model(model)
    torch.cuda.synchronize()
    setup = time.perf_counter() - start
    labels, post, times = [], [], []
    for r in rows:
        start = time.perf_counter()
        label = session.predict(torch.from_numpy(x[r:r + 1]))
        prob = session.last.detach().cpu().numpy()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        if tuple(label.shape) != (1,) or label.dtype != torch.int64 or int(label[0]) != int(prob.argmax()):
            raise ValueError("Team C returned an invalid decision")
        labels.append(int(label[0]))
        post.append(prob)
    return np.asarray(labels, np.int8), np.asarray(post), np.asarray(times), setup


def fold(data, packages, out, mode, person, smoke):
    pkg = packages / ("C1ret/pkg_ret" if mode == "ret" else f"C1/pkg_{person}")
    sources = check_sources(pkg, mode, person, data)
    available = sorted({s for s in data["stream"] if s.split("|")[0] == person} - sources)
    if not available:
        raise ValueError(f"{mode} {person}: no sessions")
    _, streaming = harness.official()
    spec = importlib.util.spec_from_file_location("team_c_fold", pkg / "submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.backends.cudnn.benchmark = False
    result = {}
    for arm in ARMS:
        model = module.load_package(pkg, "cuda", VARIANTS[arm])
        warm_rows = next(iter(harness.plans(data, ORDERS[0], available).values()))[:32]
        run(model, data["X"], warm_rows, streaming)
        for order in ORDERS[:1] if smoke else ORDERS:
            for stream, rows in harness.plans(data, order, available).items():
                visit = stream.split("|")[1]
                key = f"{arm}_{order}_{visit}"
                if smoke:
                    rows = rows[:32]
                labels, post, times, setup = run(model, data["X"], rows, streaming)
                if smoke:
                    shorter = run(model, data["X"], rows[:24], streaming)
                    np.testing.assert_array_equal(shorter[0], labels[:24])
                    np.testing.assert_allclose(shorter[1], post[:24], atol=5e-5, rtol=0)
                    print(f"  prefix max posterior drift {np.max(np.abs(shorter[1] - post[:24])):.2g}")
                result.update({f"rows_{order}_{visit}": rows, f"labels_{key}": labels,
                               f"post_{key}": post, f"secs_{key}": times, f"setup_{key}": setup})
                print(f"OK {mode} {person} {key}: {len(rows)} windows", flush=True)
    result.update(gpu=np.asarray(torch.cuda.get_device_name()),
                  package_sha=np.asarray([f"{f}:{digest(pkg / f)}"
                                          for f in ("submission.py", "train.json", "weights.pt")]))
    final = out / f"team_{mode}_{person}.npz"
    with final.with_suffix(".tmp").open("wb") as pending:
        np.savez(pending, **result)
    final.with_suffix(".tmp").rename(final)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--stock-data", type=Path, help="verify an independently archived cap250 input")
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("folds", nargs="+", help="e.g. new:P01 ret:P04")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Team C replay requires a Slurm GPU worker")
    if args.stock_data:
        check_stock_input(args.data, args.stock_data)
    args.out.mkdir(parents=True, exist_ok=True)
    data = read_input(args.data)
    for pair in args.folds:
        mode, person = pair.split(":")
        if mode not in {"new", "ret"} or person not in (harness.PEOPLE if mode == "new" else harness.RETURNING):
            raise ValueError(pair)
        fold(data, args.packages, args.out, mode, person, args.smoke)


if __name__ == "__main__":
    main()
