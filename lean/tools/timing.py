"""Paired, warm run-time comparison of two packages on the same public sessions, in one process.

Usage: python lean/tools/timing.py PKG_A PKG_B --public proteus_windows.npz [--sessions 6] [--rounds 2] [--out JSON]

Both packages first run one untimed session. Then every chosen session is streamed by both, one after the other
(the order alternates), each on a fresh copy as in the evaluator; the time covers the copy, the reset and every
``predict`` call. Reported: seconds per package, ms per window and the per-session ratio B / A.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from record import load_model, load_windows  # noqa: E402


def stream_seconds(model, X):
    t0 = time.perf_counter()
    cur = copy.deepcopy(model)
    cur.reset_state()
    for i in range(len(X)):
        cur.predict(torch.from_numpy(X[i:i + 1]))
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--public", required=True)
    ap.add_argument("--sessions", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out")
    args = ap.parse_args()
    d = load_windows(args.public)
    names = [str(c) for c in d["ch_names"]]
    models = {"a": load_model(args.a, names, "pkg_a"), "b": load_model(args.b, names, "pkg_b")}
    streams = list(dict.fromkeys(d["stream"]))
    X = {s: np.ascontiguousarray(d["X"][d["stream"] == s]) for s in streams[:args.sessions + 1]}
    for m in models.values():  # warm-up on a session that is not timed
        stream_seconds(m, X[streams[args.sessions]])
    rows = []
    for rnd in range(args.rounds):
        for i, s in enumerate(streams[:args.sessions]):
            order = ("a", "b") if (i + rnd) % 2 == 0 else ("b", "a")
            sec = {k: stream_seconds(models[k], X[s]) for k in order}
            rows.append(dict(session=s.split("/")[-1], round=rnd, first=order[0], windows=len(X[s]), **sec))
            print(f"{rows[-1]['session']:8s} round {rnd} first {order[0]}  a {sec['a']:6.2f} s  b {sec['b']:6.2f} s  "
                  f"b/a {sec['b'] / sec['a']:.3f}", flush=True)
    n = sum(r["windows"] for r in rows)
    ratio = np.array([r["b"] / r["a"] for r in rows])
    res = dict(a=str(args.a), b=str(args.b), windows=n, seconds_a=sum(r["a"] for r in rows), seconds_b=sum(r["b"] for r in rows),
               ms_per_window_a=1e3 * sum(r["a"] for r in rows) / n, ms_per_window_b=1e3 * sum(r["b"] for r in rows) / n,
               ratio_of_sums=sum(r["b"] for r in rows) / sum(r["a"] for r in rows), ratio_median=float(np.median(ratio)),
               ratio_min=float(ratio.min()), ratio_max=float(ratio.max()), runs=rows)
    print(json.dumps({k: v for k, v in res.items() if k != "runs"}, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
