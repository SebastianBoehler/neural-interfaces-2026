"""Check the exact hard-only Team B archives through their official Solver."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import torch

from build_hard_team_b import BASE, BASE_SHA, DOSES, name, once, verify_parent
from cb_speed_bench import stats, timed
from team_b_speed_bench import benchmark, check_features

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "cleanslate/harness"), str(ROOT / "2026-competition")]
from benchmark_utils.streaming import fresh_model
import data as D

DATA_SHA = "2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327"
ZERO_STREAMS = ("P01|1", "P04|2")


def unpack(archive, directory):
    with ZipFile(archive) as package:
        package.extractall(directory)
    path = directory / "submission.py"
    spec = importlib.util.spec_from_file_location("hard_team_b_solver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load(module, directory, data, device):
    return module.Solver().load_model(dict(submission_dir=directory, device=device, ch_names=data["ch_names"]))


def clean(model):
    team = model.team_b
    if model.hard_actors is not None or team.windows or team.cov is not None or team.embeddings is not None:
        raise RuntimeError("self-test or another session left state on the template")
    if any(g.graph is not None or g.failed or g.replays for g in team.feature_graphs.values()):
        raise RuntimeError("self-test or another session left a graph on the template")


def label(model, window):
    result = model.predict(window[None])
    if result.dtype != torch.int64 or result.shape != (1,) or not 0 <= int(result[0]) <= 2:
        raise ValueError(f"invalid official Solver label: {result}")
    return int(result[0])


def stream(template, windows, *, team_path=None, capture=False):
    start = perf_counter()
    model = fresh_model(template)
    if team_path == "eager":
        model.team_b.feature_graphs.clear()
    out, taps = [], []
    for window in windows:
        out.append(label(model, window))
        if capture:
            taps.append(np.r_[model.trace["v6"]["mix6"], model.trace["out"]])
    if model.dead or model.log6["err"] or model.log6["v6"] != len(windows) or model.routeB.log["err"] or model.cp.nfail:
        raise RuntimeError(f"degraded Solver: {model.log6}")
    if team_path and len(model.team_b.windows) != len(windows):
        raise RuntimeError("Team B skipped live windows")
    if team_path == "graph":
        check_features(model.team_b, len(windows))
    return np.asarray(out), np.asarray(taps), dict(wall_s=perf_counter() - start, rec=model.log6["rec"], refits=model.ft.refits)


def zero_check(base, module, directory, data, arm, smoke):
    with patch.object(module.TeamBlendStack, "TEAM_B_MIX", 0.0):
        zero = load(module, directory, data, "cpu")
    if zero.team_b is not None:
        raise RuntimeError(f"{arm}: blend zero loaded Team B")
    counts = {}
    for session, rows in D.plans(data, 71120, ZERO_STREAMS).items():
        windows = torch.from_numpy(data["X"][rows[:8] if smoke else rows])
        a = stream(base, windows, capture=True)
        b = stream(zero, windows, capture=True)
        for x, y in zip(a[:2], b[:2], strict=True):
            if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
                raise RuntimeError(f"{arm} {session}: zero labels, mixer or final scores differ")
        counts[session] = len(windows)
    if not smoke and sum(counts.values()) != 379:
        raise RuntimeError(f"wrong blend-zero coverage: {counts}")
    return counts


def check_load_fails(module, directory, data, device):
    encoder = module._TEAM_B["Encoder"]
    original = encoder.posterior
    checks = {
        "exception": patch.object(encoder, "posterior", side_effect=RuntimeError("injected Team B posterior failure")),
        "wrong_posterior": patch.object(encoder, "posterior", lambda obj, x: original(obj, x).roll(1)),
    }
    if device == "cuda:0":
        capture = module.TeamBFeatureGraph.capture

        def capture_for(size):
            def injected(forward, aligned, dev):
                if len(aligned) == size:
                    raise RuntimeError(f"injected batch {size} capture failure")
                return capture(forward, aligned, dev)
            return injected

        for size in (1, 64):
            checks[f"capture_{size}"] = patch.object(module.TeamBFeatureGraph, "capture", side_effect=capture_for(size))
    errors = {}
    for key, injection in checks.items():
        with injection:
            try:
                load(module, directory, data, device)
            except RuntimeError as exc:
                if "load-time self-test: Team B" not in str(exc):
                    raise
                errors[key] = str(exc)
            else:
                raise RuntimeError(f"Team B {key} injection loaded successfully")
    return errors


def check_dead(template, windows):
    model = fresh_model(template)
    with patch.object(model.team_b, "posterior", side_effect=RuntimeError("injected live Team B failure")) as team:
        labels = [label(model, window) for window in windows]
    if not model.dead or model.log6["err"] != 1 or team.call_count != 1 or model.hard_window is not None:
        raise RuntimeError(f"dead-session guard failed: {model.log6}")
    return dict(windows=len(labels), errors=model.log6["err"], dead=model.dead)


def plans(data, order):
    sessions = tuple(dict.fromkeys(data["stream"]))
    if order == "recording":
        return {s: np.flatnonzero(data["stream"] == s) for s in sessions}
    return D.plans(data, int(order), sessions)


def compare(model, data, device, order, *, part=0, count=1, smoke=False):
    reports = {}
    for i, (session, rows) in enumerate(list(plans(data, order).items())[part::count]):
        if smoke and i:
            break
        windows = torch.from_numpy(data["X"][rows[:72] if smoke else rows]).to(device)
        paths = ("eager", "graph") if i % 2 == 0 else ("graph", "eager")
        results = {path: stream(model, windows, team_path=path) for path in paths}
        eager, graph = results["eager"], results["graph"]
        changes = np.flatnonzero(eager[0] != graph[0]).tolist()
        reports[session] = dict(windows=len(windows), changed=changes, eager=eager[2], graph=graph[2])
        print(order, session, reports[session], flush=True)
        if changes or any(eager[2][key] != graph[2][key] for key in ("rec", "refits")):
            raise RuntimeError(f"{session}: graph and eager differ")
    clean(model)
    return reports


def timing(model, data):
    windows = data["X"]
    plan = plans(data, "71000")
    eager = fresh_model(model)
    reset = eager.team_b.reset

    def reset_eager():
        reset()
        eager.team_b.feature_graphs.clear()

    with patch.object(eager.team_b, "reset", reset_eager):
        team = benchmark([eager, model], windows, plan, graphed=True)
    for i in next(iter(plan.values()))[:100]:
        model.hard_probability(windows[i])
    hard_ms, warm_ms = [], []
    for rows in plan.values():
        model.hard_actors = None
        for age, row in enumerate(rows):
            _, ms = timed(model.hard_probability, windows[row])
            hard_ms.append(ms)
            if age:
                warm_ms.append(ms)
    total = stats(hard_ms)["mean_ms"] + team["candidate"]["session"]["mean_ms"]
    return dict(hard=stats(hard_ms), hard_warm=stats(warm_ms), team=team, added_ms=total,
                projected_s=2555 + 20083 * total / 1000)


def run(args):
    verify_parent(args.data, DATA_SHA)
    archive = args.zips / name(args.arm)
    verify_parent(archive, args.sha)
    data = D.load(args.data)
    if data["X"].shape != (2655, 41, 480) or len(set(data["stream"])) != 14:
        raise ValueError("unexpected public data coverage")
    device = "cpu" if args.stage in ("local", "zero") else "cuda:0"
    with TemporaryDirectory() as scratch:
        directory = Path(scratch) / args.arm
        module = unpack(archive, directory)
        report = dict(stage=args.stage, archive_sha256=args.sha, torch=torch.__version__)
        if device == "cuda:0":
            report["gpu"] = torch.cuda.get_device_name()
            expected = os.environ.get("NI26_EXPECT_GPU", "A100")  # set to "2080 Ti" for the Galvani RTX runs
            if expected not in report["gpu"]:
                raise RuntimeError(f"expected {expected}, got {report['gpu']}")
        if args.stage in ("local", "zero"):
            base_zip = args.zips / f"{BASE}.zip"
            verify_parent(base_zip, BASE_SHA)
            base_dir = Path(scratch) / "base"
            base = load(unpack(base_zip, base_dir), base_dir, data, device)
            report["zero"] = zero_check(base, module, directory, data, args.arm, args.stage == "local")
        if args.stage != "zero":
            model = load(module, directory, data, device)
            if model.TEAM_B_MIX != DOSES[args.arm] or len(model.team_b.nets) != 9:
                raise RuntimeError("unexpected Team B dose or member count")
            clean(model)
        if args.stage in ("local", "smoke"):
            report["load_errors"] = check_load_fails(module, directory, data, device)
            rows = plans(data, "71000")[ZERO_STREAMS[0]][:8]
            report["dead_guard"] = check_dead(model, torch.from_numpy(data["X"][rows]).to(device))
        if args.stage == "smoke":
            report["orders"] = {order: compare(model, data, device, order, smoke=True)
                                for order in ("recording", "71000")}
        if args.stage == "terminal":
            report.update(order=args.order, sessions=compare(model, data, device, args.order, part=args.part, count=2))
        if args.stage == "timing":
            report.update(timing(model, data))
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(args.stage, args.arm, "PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("local", "zero", "smoke", "terminal", "timing"))
    parser.add_argument("arm", choices=tuple(DOSES))
    parser.add_argument("zips", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--order", choices=("recording", "71000"), default="71000")
    parser.add_argument("--part", choices=(0, 1), type=int, default=0)
    run(parser.parse_args())
