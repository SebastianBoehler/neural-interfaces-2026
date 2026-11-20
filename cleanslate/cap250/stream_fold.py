"""Run the official clean-slate stream and reject incomplete or degraded cap250 runs."""

import argparse
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

from source import ROOT, default_data

sys.path.insert(0, str(ROOT / "cleanslate/harness"))
import data as harness


def stream(args):
    data = harness.load(args.data)
    sessions = sorted(set(data["session"][data["person"] == args.held]))
    if args.sessions != "all":
        requested = args.sessions.split(",")
        if set(requested) - set(sessions):
            raise ValueError("requested visit is missing from official replay rows")
        sessions = requested
    orders = [int(order) for order in args.orders.split(",")]
    streams = [f"{args.held}|{session}" for session in sessions]
    if not streams or args.out.exists():
        raise ValueError("missing held rows or existing stream output")
    with TemporaryDirectory(prefix=".cap250-stream-", dir=args.out.parent) as stage:
        pending = Path(stage) / "predictions.npz"
        cmd = [sys.executable, "-B", str(ROOT / "cleanslate/harness/stream.py"), "--pkg", str(args.pkg),
               "--held", args.held, "--sessions", args.sessions, "--orders", args.orders,
               "--data", str(args.data), "--device", "cuda:0", "--out", str(pending)]
        result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
        sys.stderr.write(result.stderr)
        result.check_returncode()
        errors = re.findall(r"\[v6\].* errors (\d+)", result.stderr)
        if len(errors) != len(streams) * len(orders) or any(int(n) for n in errors):
            raise ValueError("the cap250 stream missed a session status or had a v6 error")
        with np.load(pending, allow_pickle=False) as predictions:
            for order in orders:
                plan = harness.plans(data, order, streams)
                for session, expected in zip(sessions, plan.values(), strict=True):
                    if not np.array_equal(predictions[f"rows_{order}_{session}"], expected):
                        raise ValueError(f"delivery disagreement for {args.held}|{session}, order {order}")
        pending.rename(args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkg", type=Path, required=True)
    parser.add_argument("--held", required=True)
    parser.add_argument("--sessions", default="all")
    parser.add_argument("--orders", required=True)
    parser.add_argument("--data", type=Path, default=default_data())
    parser.add_argument("--out", type=Path, required=True)
    stream(parser.parse_args())
