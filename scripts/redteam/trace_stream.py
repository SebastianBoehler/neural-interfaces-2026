"""Run the pinned clean-fold Solver stream and capture its post-predict counters."""

import argparse
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


def stream(args):
    sys.path.insert(0, str(Path(os.environ["CAP250_RELEASE"]) / "cleanslate/harness"))
    import stream as pinned

    data = pinned.D.load(args.data)
    sessions = sorted(set(data["session"][data["person"] == args.held]))
    if args.sessions != "all":
        requested = args.sessions.split(",")
        if set(requested) - set(sessions):
            raise ValueError(f"missing visit for {args.held}: {requested}")
        sessions = [s for s in sessions if s in requested]
    streams = [f"{args.held}|{s}" for s in sessions]
    orders = [int(order) for order in args.orders.split(",")]
    if not streams or args.out.exists():
        raise ValueError("missing held rows or existing stream output")

    records = []
    original_load = pinned.load_solver

    def load(*load_args):
        model = original_load(*load_args)
        predict = type(model).predict

        def observed(m, x):
            before_ft = m.ft.refits
            before_cp = m.cp.fused
            output = predict(m, x)
            if m.dead or m.log6["err"] or m.routeB.log["err"] or m.cp.nfail:
                raise RuntimeError(f"degraded solver path for {args.arm}, {args.held}")
            records.append((m.log6["n"] - 1, m.log6["rec"], m.routeB.log["routed"],
                            m.ft.refits - before_ft, m.cp.fused - before_cp))
            return output

        type(model).predict = observed
        print(f"[rt] {args.arm} {type(model).__name__} {type(model.ft).__name__}", flush=True)
        return model

    pinned.load_solver = load
    with TemporaryDirectory(prefix=".rt-stream-", dir=args.out.parent) as tmp:
        pending = Path(tmp) / "pred.npz"
        sys.argv = ["stream.py", "--pkg", args.pkg, "--held", args.held,
                    "--sessions", args.sessions, "--orders", args.orders, "--data", args.data,
                    "--device", args.device, "--out", str(pending)]
        pinned.main()
        output = {}
        offset = 0
        with np.load(pending, allow_pickle=False) as pred:
            output.update((k, pred[k]) for k in pred.files)
            for order in orders:
                for stream_id, expected in pinned.D.plans(data, order, streams).items():
                    visit = stream_id.split("|")[1]
                    key = f"{order}_{visit}"
                    if not np.array_equal(pred[f"rows_{key}"], expected):
                        raise ValueError(f"delivery disagreement: {args.held}|{visit}, {order}")
                    age, rec, total_routes, ft, cp = np.asarray(records[offset:offset + len(expected)]).T
                    offset += len(expected)
                    np.testing.assert_array_equal(age, np.arange(len(expected)))
                    routed = np.diff(total_routes, prepend=0).astype(bool)
                    late_ft = age[(rec >= 0) & (age > rec) & (ft > 0)]
                    late_cp = age[(rec >= 0) & (age > rec) & (cp > 0)]
                    if args.arm in ("R1a", "R2", "R4") and not len(late_ft):
                        raise ValueError(f"{args.arm} {key}: no post-recognition refit")
                    if args.arm == "R1a":
                        np.testing.assert_array_equal(late_ft, [100])
                    if args.arm == "R5" and len(late_ft):
                        raise ValueError(f"{args.arm} {key}: late refit with FT retired")
                    if args.arm in ("R5", "R2", "R4") and not len(late_cp):
                        raise ValueError(f"{args.arm} {key}: no post-recognition fusion")
                    print(f"[rt] {key} recognised_at {rec[-1]} routed {routed.sum()} "
                          f"late_refits {len(late_ft)} late_fused {len(late_cp)}", flush=True)
                    output[f"routed_{key}"] = routed
                    output[f"recognised_{key}"] = rec[-1]
        if offset != len(records):
            raise ValueError("trace coverage differs from the requested deliveries")
        np.savez(pending, **output)
        pending.rename(args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--pkg", required=True)
    parser.add_argument("--held", required=True)
    parser.add_argument("--sessions", required=True)
    parser.add_argument("--orders", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--out", type=Path, required=True)
    stream(parser.parse_args())
