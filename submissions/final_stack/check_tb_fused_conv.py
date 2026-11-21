"""Exercise the signed fused Team B archive and its two live CUDA capture failure paths."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import warnings

import numpy as np
import torch

import check_hard_team_b as checks
from build_hard_team_b import verify_parent


def check_fallback(template, windows):
    labels, taps, baseline = checks.stream(template, windows, team_path="graph", capture=True)
    model = checks.fresh_model(template)
    states = model.team_b.feature_graphs
    with warnings.catch_warnings(record=True) as alerts, ExitStack() as stack:
        warnings.simplefilter("always", RuntimeWarning)
        faults = {
            size: stack.enter_context(patch.object(state, "capture", side_effect=RuntimeError(
                f"injected fused Team B capture failure for batch {size}")))
            for size, state in states.items()
        }
        got_labels, got_taps = [], []
        for window in windows:
            got_labels.append(checks.label(model, window))
            got_taps.append(np.r_[model.trace["v6"]["mix6"], model.trace["out"]])

    if model.dead or model.log6["err"] or model.log6["v6"] != len(windows):
        raise RuntimeError(f"capture fault degraded Solver: {model.log6}")
    if len(model.team_b.windows) != len(windows) or model.routeB.log["err"] or model.cp.nfail:
        raise RuntimeError("capture fault skipped Team B or degraded the stream")
    if baseline["rec"] != model.log6["rec"] or baseline["refits"] != model.ft.refits:
        raise RuntimeError("capture fault changed adaptation")
    counts = {size: fault.call_count for size, fault in faults.items()}
    if counts != {1: 1, 64: 1} or any(not g.failed or g.replays or g.graph is not None for g in states.values()):
        raise RuntimeError(f"capture faults did not select eager once per shape: {counts}")
    messages = [str(alert.message) for alert in alerts if "using eager: injected fused Team B" in str(alert.message)]
    if len(messages) != 2:
        raise RuntimeError(f"expected two capture warnings, got {messages}")
    np.testing.assert_array_equal(got_labels, labels)
    np.testing.assert_allclose(got_taps, taps, rtol=0, atol=1e-6)
    return dict(windows=len(windows), changed_labels=0, max_abs=float(np.max(np.abs(got_taps - taps))),
                faults=counts, warnings=messages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("sha256")
    parser.add_argument("data", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    verify_parent(args.archive, args.sha256)
    verify_parent(args.data, checks.DATA_SHA)
    data = checks.D.load(args.data)
    device = "cuda:0"
    rows = next(iter(checks.plans(data, "71000").values()))[:72]
    windows = torch.from_numpy(data["X"][rows]).to(device)
    with TemporaryDirectory() as temp:
        directory = Path(temp)
        module = checks.unpack(args.archive, directory)
        model = checks.load(module, directory, data, device)
        checks.clean(model)
        fused = [net.body.fused for net in model.team_b.nets if getattr(net.body, "fused", None) is not None]
        if len(fused) != 6 or not all(t.is_cuda for pair in fused for t in pair):
            raise RuntimeError("expected all six fused CUDA bodies")
        fallback = check_fallback(model, windows)
        dead = checks.check_dead(model, windows[:8])
        checks.clean(model)
        load_guards = checks.check_load_fails(module, directory, data, device) if hasattr(model, "_check_team_b") else None
    report = dict(archive_sha256=args.sha256, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  fused_bodies=len(fused), fallback=fallback, dead_session=dead, team_b_load_guards=load_guards)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
