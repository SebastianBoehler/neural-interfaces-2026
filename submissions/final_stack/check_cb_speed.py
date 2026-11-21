"""Qualify the fast hardcb ZIP on real PROTEUS windows through the Solver."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch
import warnings
from zipfile import ZipFile

import braindecode
import numpy as np
import torch

from build_cb_speed import NAME, digest, verify_pair
from build_team_b_speed import verify as verify_team
from cb_speed_bench import ATOL, benchmark, check_graph
import team_b_speed_bench

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "cleanslate/harness"), str(ROOT / "2026-competition")]
from benchmark_utils.streaming import fresh_model
from data import delivery
from stream import load_solver

DATA_SHA = "2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327"


def load_models(stack, zips, channels, device, team_b=None):
    if team_b:
        paths = verify_team(zips, team_b == "graph")
    else:
        paths = zips / f"{NAME}.zip", zips / f"{NAME}_cbgraph.zip"
        verify_pair(*paths)
    models = []
    for path in paths:
        directory = Path(stack.enter_context(TemporaryDirectory()))
        with ZipFile(path) as archive:
            archive.extractall(directory)
        models.append(load_solver(directory, channels, device))
    expected = "TeamBlendStack" if team_b else "StackEnsemble"
    if any(type(m).__name__ != expected or len(m.cb.nets) != 3 for m in models):
        raise RuntimeError("expected the full three-seed hardcb Solver")
    if team_b and any(m.TEAM_B_MIX != 0.25 or len(m.team_b.nets) != 9 for m in models):
        raise RuntimeError("expected the nine25 Team B Solver")
    graph = models[1].cb._forward_graph
    if graph.graph is not None or graph.failed or graph.replays or models[1].cb.n:
        raise RuntimeError("constructor self-test leaked CBraMod state")
    return models, [digest(p) for p in paths]


def compare_labels(models, windows, device, check_active, team_b):
    labels, timings, traces = [], [], []
    x = torch.from_numpy(windows).to(device)
    for i, template in enumerate(models):
        start = perf_counter()
        model = fresh_model(template)
        output = []
        for row in x:
            pred = model.predict(row[None])
            if pred.shape != (1,) or pred.dtype != torch.int64 or pred.item() not in (0, 1, 2):
                raise RuntimeError(f"invalid terminal label: {pred}")
            output.append(pred.item())
        timings.append(perf_counter() - start)
        if model.dead or model.log6["err"] or model.routeB.log["err"] or model.cp.nfail:
            raise RuntimeError(f"degraded Solver arm {i}: {model.log6}")
        if model.cb.n != len(x):
            raise RuntimeError(f"arm {i} invoked CBraMod {model.cb.n} times for {len(x)} windows")
        if i == 1 and check_active:
            check_graph(model.cb, len(x))
        if team_b:
            if len(model.team_b.windows) != len(x):
                raise RuntimeError(f"arm {i} skipped Team B inference")
            if i == 1 and check_active and team_b == "graph":
                team_b_speed_bench.check_features(model.team_b, len(x))
        traces.append(dict(rec=model.log6["rec"], refits=model.ft.refits))
        labels.append(np.array(output))
        del model
    changed = np.flatnonzero(labels[0] != labels[1])
    return dict(windows=len(x), changed=changed.tolist(), traces=traces, wall_s=timings)


def check_paths(models, windows):
    original, fast = (model.cb for model in models)
    original.reset()
    fast.reset()
    truth = np.stack([original.prob(x) for x in windows])
    first = fast.prob(windows[0])
    # Once captured, replay must keep producing the right results with eager forward disabled.
    with patch.object(fast, "_forward_seeds", side_effect=AssertionError("unexpected eager forward")):
        got = np.stack([first, *(fast.prob(x) for x in windows[1:])])
    check_graph(fast, len(windows))
    clone = fresh_model(models[1])
    if clone.cb._forward_graph.graph is not None or clone.cb.n:
        raise RuntimeError("fresh_model carried graph or prefix state across a session")
    np.testing.assert_allclose(clone.cb.prob(windows[0]), first, rtol=0, atol=ATOL)
    check_graph(clone.cb, 1)
    del clone
    fast.reset()
    with warnings.catch_warnings(record=True) as alerts:
        warnings.simplefilter("always", RuntimeWarning)
        with patch.object(torch.cuda, "CUDAGraph", side_effect=RuntimeError("injected capture failure")) as factory:
            fallback = np.stack([fast.prob(x) for x in windows])
        if factory.call_count != 1 or not fast._forward_graph.failed or fast._forward_graph.replays:
            raise RuntimeError("expected eager fallback after one failed capture attempt")
    if not any("using eager: injected capture failure" in str(alert.message) for alert in alerts):
        raise RuntimeError("capture failure was silent")
    np.testing.assert_allclose(got, truth, rtol=0, atol=ATOL)
    np.testing.assert_allclose(fallback, truth, rtol=0, atol=ATOL)
    fast.reset()
    return dict(graph_max_abs=float(np.max(np.abs(got - truth))),
                forced_eager_max_abs=float(np.max(np.abs(fallback - truth))))


def plans(sessions, order):
    rows = np.arange(len(sessions)) if order == "recording" else delivery(sessions, int(order))
    return {s: rows[sessions[rows] == s] for s in dict.fromkeys(sessions)}


def run(args):
    if digest(args.data) != DATA_SHA:
        raise ValueError("PROTEUS export digest differs")
    if args.device == "cpu" and args.stage != "smoke":
        raise ValueError("full qualification needs CUDA")
    with np.load(args.data, allow_pickle=False) as data:
        sessions = np.char.add(np.char.add(data["subject"], "|"), data["session"])
        channels, x = data["ch_names"], data["X"]
    if x.shape != (2655, 41, 480) or len(set(sessions)) != 14:
        raise ValueError(f"unexpected public data coverage: {x.shape}")
    order = plans(sessions, args.order)
    with ExitStack() as stack:
        models, hashes = load_models(stack, args.zips, channels, args.device, args.team_b)
        cuda = args.device != "cpu"
        result = dict(stage=args.stage, order=args.order, team_b=args.team_b, zip_sha256=hashes, torch=torch.__version__,
                      braindecode=braindecode.__version__, cuda=torch.version.cuda,
                      gpu=torch.cuda.get_device_name() if cuda else None,
                      shared_prior=bool(models[0].cb.shared))
        if args.stage == "smoke":
            if cuda:
                result["paths"] = check_paths(models, x[next(iter(order.values()))[:8]])
                result["bench_smoke"] = benchmark(models, x, order, smoke=True)
                if args.team_b:
                    graphed = args.team_b == "graph"
                    if graphed:
                        selected = x[next(iter(order.values()))[:72]]
                        result["team_b_forced_eager_abs"] = team_b_speed_bench.check_fallback(models[1].team_b, selected)
                    result["team_b_bench"] = team_b_speed_bench.benchmark(models, x, order, graphed=graphed, smoke=True)
            result["sessions"] = {
                label: compare_labels(models, x[next(iter(plans(sessions, label).values()))[:8]], args.device, cuda, args.team_b)
                for label in ("recording", "71000")
            }
        elif args.stage == "bench":
            result.update(benchmark(models, x, order))
            if args.team_b:
                result["team_b_bench"] = team_b_speed_bench.benchmark(models, x, order, graphed=args.team_b == "graph")
        else:
            part, count = args.part
            result["sessions"] = {}
            for stream, rows in list(order.items())[part::count]:
                info = compare_labels(models, x[rows], args.device, cuda, args.team_b)
                print(stream, json.dumps(info), flush=True)
                result["sessions"][stream] = info
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    if any(s["changed"] or s["traces"][0] != s["traces"][1] for s in result.get("sessions", {}).values()):
        raise RuntimeError("terminal labels or adaptation traces differ")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("smoke", "bench", "terminal"))
    p.add_argument("zips", type=Path)
    p.add_argument("data", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    p.add_argument("--team-b", choices=("cb", "graph"))
    p.add_argument("--order", choices=("recording", "71000"), default="71000")
    p.add_argument("--part", nargs=2, type=int, default=(0, 1), metavar=("INDEX", "COUNT"))
    args = p.parse_args()
    if not 0 <= args.part[0] < args.part[1] <= 14:
        p.error("choose a valid zero-based session partition")
    run(args)
