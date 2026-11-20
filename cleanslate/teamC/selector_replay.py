"""Validate paired fold streams and freeze selector decisions before reading targets."""

import argparse
import json
from pathlib import Path
import re
import time

import numpy as np

from selector import RULES, Selector
from selector_stream import ARMS, ORDERS, digest, harness, read_input


def stock_status(work, mode, person):
    key = f"stock.{mode}_{person}.stream"
    receipts = [json.loads(line) for line in (work / "jobs.jsonl").read_text().splitlines()]
    job, = {receipt["job"] for receipt in receipts if receipt["node"] == key}
    log, = (work / "logs").glob(f"*-{job}_*.out")
    contents = log.read_text()
    passes = re.findall(r"^PASS (P\d{2}) (\d+) (\w+) (\d+) [\d.]+s$", contents, re.M)
    status = re.findall(r"^\[v6\] windows (\d+) .*?recognised_at (-?\d+) .*?errors (\d+)$", contents, re.M)
    if len(passes) != len(status):
        raise ValueError(f"{key}: missing completion or recognition status")
    result = {}
    for (p, order, visit, n), (seen, at, errors) in zip(passes, status, strict=True):
        if p != person or n != seen or int(errors) != 0 or not -1 <= int(at) < int(n):
            raise ValueError(f"{key}: invalid session status")
        result[int(order), visit] = int(at)
    if len(result) != len(passes):
        raise ValueError(f"{key}: duplicate session status")
    return result, log


def replay(x, cap, team, post):
    selector = Selector()
    pred, duration = [], []
    for window, a, b, p in zip(x, cap, team, post, strict=True):
        start = time.perf_counter()
        pred.append(selector.step(window, a, b, p))
        duration.append(time.perf_counter() - start)
    return np.asarray(pred), np.asarray(duration)


def compare(cap, team, post, x, at, smoke):
    if not (np.isin(cap, (0, 1, 2)).all() and post.shape == (len(cap), 3)
            and np.isfinite(post).all() and (post >= 0).all()
            and np.allclose(post.sum(axis=1), 1.0, atol=5e-6)
            and np.array_equal(team, post.argmax(axis=1))):
        raise ValueError("invalid paired predictions")
    n = len(cap)
    if smoke:
        zero = Selector()
        blind = [zero.step(w, a, b, p, enabled=False)
                 for w, a, b, p in zip(x, cap, team, post, strict=True)]
        np.testing.assert_array_equal(blind, np.repeat(cap[:, None], len(RULES), axis=1))
    pred, times = replay(x, cap, team, post)
    np.testing.assert_array_equal(pred[cap == team], np.repeat(cap[cap == team, None], len(RULES), axis=1))
    if smoke:
        end = n - 8
        short, _ = replay(x[:end], cap[:end], team[:end], post[:end])
        np.testing.assert_array_equal(short, pred[:end])
    active = np.arange(n) < at if at >= 0 else np.ones(n, dtype=bool)
    chosen = np.where(active[:, None], pred, cap[:, None])
    np.testing.assert_array_equal(chosen[~active], np.repeat(cap[~active, None], len(RULES), axis=1))
    return chosen, active, pred[:, 0], times


def fold(data, work, team_dir, mode, person, smoke):
    stock = work / "stock" / f"pred_{mode}_{person}.npz"
    team = team_dir / f"team_{mode}_{person}.npz"
    status, log = stock_status(work, mode, person)
    streams = sorted({s for s in data["stream"] if s.split("|")[0] == person
                      and (mode == "new" or not s.endswith("|1"))})
    expected = {(o, s.split("|")[1]) for o in ORDERS for s in streams}
    if status.keys() != expected:
        raise ValueError(f"{stock}: order or visit set differs")
    out = {"source_sha": np.array([digest(f) for f in (stock, team, log)])}
    spec = {}
    with np.load(stock, allow_pickle=False) as a, np.load(team, allow_pickle=False) as b:
        out.update(gpu=b["gpu"], package_sha=b["package_sha"])
        for order in ORDERS[:1] if smoke else ORDERS:
            for stream, rows in harness.plans(data, order, streams).items():
                visit = stream.split("|")[1]
                suffix = f"{order}_{visit}"
                rows_key = f"rows_{suffix}"
                np.testing.assert_array_equal(a[rows_key], rows)
                if smoke:
                    rows = rows[:32]
                np.testing.assert_array_equal(b[rows_key], rows)
                out[rows_key] = rows
                cap = a[f"labels_{suffix}"][:len(rows)]
                out[f"cap_{suffix}"] = cap
                spec[suffix] = status[order, visit]
                for arm in ARMS:
                    key = f"{arm}_{suffix}"
                    team_pred = b[f"labels_{key}"]
                    chosen, active, ungated, time = compare(
                        cap, team_pred, b[f"post_{key}"], data["X"][rows], status[order, visit], smoke
                    )
                    out.update({f"team_{key}": team_pred, f"chosen_{key}": chosen, f"active_{key}": active,
                                f"ungated_{key}": ungated, f"selector_secs_{key}": time,
                                f"team_secs_{key}": b[f"secs_{key}"], f"setup_{key}": b[f"setup_{key}"]})
    out["recognition"] = np.array(json.dumps(spec, sort_keys=True))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    data = read_input(args.data)
    cohorts = [("new", harness.PEOPLE), ("ret", harness.RETURNING)]
    subdir = "smoke" if args.smoke else "full"
    out = args.work / f"blind_{subdir}"
    out.mkdir(exist_ok=False)
    for mode, people in cohorts:
        if args.smoke:
            people = ["P04"] if mode == "new" else ["P10"]
        for person in people:
            result = fold(data, args.work, args.work / subdir, mode, person, args.smoke)
            path = out / f"{mode}_{person}.npz"
            np.savez(path, **result)
            print(f"OK {mode} {person} {digest(path)}", flush=True)


if __name__ == "__main__":
    main()
