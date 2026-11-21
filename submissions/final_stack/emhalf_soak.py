"""Replay signed ZIPs on one fixed, unlabeled PROTEUS workload."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import resource
import socket
from hashlib import file_digest, sha256
from tempfile import TemporaryDirectory
from time import perf_counter, time
import traceback
from zipfile import ZipFile

import numpy as np
import torch

from benchmark_utils.streaming import fresh_model
from data import plans

TOTAL = 20_083
ORDER = 71000
MODEL_SEED = 42
DATA_SHA = "2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327"
PLAN_SHA = "aefaff8ad2727486c945a5db95948ae8bc6e9386625ea2fa40420771b18e45c3"
FIRST = ("P04|2", "P10|2")
RETURNING = {*FIRST, "P04|3", "P10|3"}
EM_HALF = (
    "final_stack_hard_safe_st_team_b_ba_six_fc_fp16_turing_recognised_em_half",
    "833a28721b0de0fa34f94de0328284ca9643cbaccc2dd343926f19c254949a4d",
)
ARMS = {
    "smoke": EM_HALF,
    "em_half_a": EM_HALF,
    "em_half_b": EM_HALF,
    "cap250": ("v66_cp1_cap250", "9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed"),
    "nine25_fc": (
        "final_stack_hard_safe_st_team_b_nine25_tbgraph_fc",
        "7017bd5c8a707e7233fe6b1e2df4376c28d49f6613174ec82c0fdbf700bc3f07",
    ),
}


def verify(path, expected):
    with path.open("rb") as handle:
        actual = file_digest(handle, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"{path}: expected SHA-256 {expected}, got {actual}")
    return actual


def workload(path):
    verify(path, DATA_SHA)
    with np.load(path, allow_pickle=False) as data:
        names, waves = data["ch_names"], data["X"]
        people = [s.rsplit("/", 1)[-1] for s in data["subject"]]
        streams = np.char.add(np.char.add(people, "|"), data["session"])
    keys = tuple(dict.fromkeys(streams))
    if waves.shape != (2655, 41, 480) or waves.dtype != np.float32 or len(keys) != 14:
        raise ValueError(f"unexpected public data coverage: {waves.shape}, {waves.dtype}, {keys}")
    shuffled = plans({"stream": streams}, ORDER, keys)
    order = (*FIRST, *(s for s in keys if s not in FIRST))
    schedule = []
    remaining = TOTAL
    while remaining:
        for session in order:
            rows = shuffled[session][:remaining].tolist()
            schedule.append((session, rows))
            remaining -= len(rows)
            if remaining == 0:
                break
    digest = sha256(json.dumps(schedule, separators=(",", ":")).encode()).hexdigest()
    if digest != PLAN_SHA:
        raise ValueError(f"delivery plan differs from independent pin: {digest}")
    return names, waves, schedule


def load_solver(archive, directory, names):
    with ZipFile(archive) as package:
        package.extractall(directory)
    spec = importlib.util.spec_from_file_location("soak_solver", directory / "submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    started = perf_counter()
    model = module.Solver().load_model(dict(
        submission_dir=directory, ch_names=names, device="cuda:0", n_chans=41, n_times=480, sfreq=120,
    ))
    torch.cuda.synchronize()
    return model, perf_counter() - started


def check_window(model, window, age):
    label = model.predict(window)
    if label.dtype != torch.long or label.shape != (1,) or int(label[0]) not in (0, 1, 2):
        raise ValueError(f"invalid official Solver output: {label}")
    log = model.log6
    if model.dead or log["err"] or log["v6"] != age:
        raise RuntimeError(f"v6 degraded: {log}")
    for name in ("routeB", "cf", "cp"):
        log = getattr(model, name).log
        if log["err"]:
            raise RuntimeError(f"{name} degraded: {log}")
    if model.cp.nfail:
        raise RuntimeError(f"CSP refit failed with single class: {model.cp.nfail}")
    team = getattr(model, "team_b", None)
    if team is not None:
        for size, graph in team.feature_graphs.items():
            if graph.failed:
                raise RuntimeError(f"Team B CUDA graph failed for batch {size}; see preceding archive warning")


def tenths(timings, length):
    edges = [i * length // 10 for i in range(11)]
    return [round(1000 * float(np.mean(timings[a:b])), 6) if len(timings) >= b > a else None
            for a, b in zip(edges[:-1], edges[1:], strict=True)]


def snapshot(path, state, timings, started):
    elapsed = perf_counter() - started
    n = len(timings)
    state.update(
        windows=n, compute_s=round(sum(timings), 6), wall_s=round(elapsed, 6),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        tenth_ms=tenths(timings, state["target_windows"]),
        prefix_tenth_ms=tenths(timings, n),
    )
    points = state["compute_at_windows"]
    if n % 500 == 0 or state["status"] != "running":
        points[str(n)] = state["compute_s"]
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, separators=(",", ":")) + "\n")
    tmp.replace(path)
    print(json.dumps({k: state[k] for k in (
        "arm", "status", "windows", "compute_s", "wall_s", "recognized_sessions",
        "peak_reserved_bytes", "peak_allocated_bytes", "peak_rss_bytes",
    )}, separators=(",", ":")), flush=True)


def run(args):
    started = perf_counter()
    state = dict(arm=args.arm, status="running", node=socket.gethostname(), torch=torch.__version__,
                 job_id=os.environ.get("SLURM_JOB_ID"), rank=os.environ.get("SLURM_PROCID"),
                 source_commit=args.source_commit, data_sha256=DATA_SHA, plan_sha256=PLAN_SHA,
                 order_seed=ORDER, model_seed=MODEL_SEED, target_windows=TOTAL,
                 recognized_sessions=0, recognized_returning_sessions=0, sessions=[], compute_at_windows={})
    positions = dict(stage="load", window=None, session_index=None, session=None, session_window=None)
    timings = []
    name, expected_sha = ARMS[args.arm]
    archive = args.zips / f"{name}.zip"
    try:
        random.seed(MODEL_SEED)
        np.random.seed(MODEL_SEED)
        torch.manual_seed(MODEL_SEED)
        state["gpu"] = torch.cuda.get_device_name()
        if "A100" not in state["gpu"] or torch.cuda.device_count() != 1:
            raise RuntimeError(f"expected one A100, got {state['gpu']} × {torch.cuda.device_count()}")
        state["archive_sha256"] = verify(archive, expected_sha)
        names, waves, schedule = workload(args.data)
        if args.arm == "smoke":
            schedule = schedule[:2]
            state["target_windows"] = sum(len(rows) for _, rows in schedule)
        with TemporaryDirectory(dir=os.environ.get("SLURM_TMPDIR")) as scratch:
            template, state["load_s"] = load_solver(archive, Path(scratch), names)
            snapshot(args.out, state, timings, started)
            for index, (session, rows) in enumerate(schedule, 1):
                positions.update(stage="reset", window=len(timings) + 1, session_index=index,
                                 session=session, session_window=1)
                if time() >= args.cutoff:
                    state["status"] = "partial"
                    break
                session_start = perf_counter()
                model = fresh_model(template)
                if model.log6["n"] or model.log6["rec"] != -1:
                    raise RuntimeError(f"dirty session after official cold reset: {model.log6}")
                windows = torch.from_numpy(waves[rows]).to("cuda:0")
                for age, window in enumerate(windows, 1):
                    positions.update(stage="predict", window=len(timings) + 1, session_window=age)
                    start = session_start if age == 1 else perf_counter()
                    check_window(model, window[None], age)
                    if model.log6["rec"] == age - 1:
                        state["recognized_sessions"] += 1
                        state["recognized_returning_sessions"] += session in RETURNING
                        print(f"recognition session={index} {session} window={positions['window']} age={age}", flush=True)
                    timings.append(perf_counter() - start)
                    if len(timings) % 500 == 0 and age < len(rows):
                        state["active_session"] = dict(index=index, session=session, windows=age,
                                                       recognized_at=model.log6["rec"])
                        snapshot(args.out, state, timings, started)
                    if age % 10 == 0 and time() >= args.cutoff:
                        state["status"] = "partial"
                        break
                rec = model.log6["rec"]
                cleanup = perf_counter()
                del model, windows, window
                torch.cuda.synchronize()
                timings[-1] += perf_counter() - cleanup
                state.pop("active_session", None)
                state["sessions"].append(dict(session=session, index=index, windows=age, planned=len(rows),
                                              compute_s=round(sum(timings[-age:]), 6), recognized_at=rec))
                snapshot(args.out, state, timings, started)
                if state["status"] == "partial":
                    break
            if args.arm == "smoke" and not state["recognized_returning_sessions"]:
                raise RuntimeError("smoke did not exercise returning-person recognition")
            if args.arm == "smoke" and len(timings) != state["target_windows"]:
                raise RuntimeError("smoke was cut short")
        state["status"] = "finished" if len(timings) == state["target_windows"] else "partial"
        snapshot(args.out, state, timings, started)
    except Exception as exc:
        state.update(status="error", error=str(exc), error_at=positions, traceback=traceback.format_exc())
        snapshot(args.out, state, timings, started)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=ARMS)
    parser.add_argument("zips", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--cutoff", required=True, type=int, help="absolute Unix time")
    parser.add_argument("--source-commit", required=True)
    run(parser.parse_args())
