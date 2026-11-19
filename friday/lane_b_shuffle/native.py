"""Run exact cap250 Solver calls and observe, or independently replay, their terminal stages."""

from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import torch

from make import ROOT, STOCK_SHA, digest


OFFICIAL = Path("/official") if Path("/official").is_dir() else ROOT / "2026-competition"
sys.path.insert(0, str(OFFICIAL))
from benchmark_utils.streaming import fresh_model  # noqa: E402


@contextmanager
def open_package(archive):
    with TemporaryDirectory() as temp:
        directory = Path(temp)
        with ZipFile(archive) as bundle:
            bundle.extractall(directory)
        spec = importlib.util.spec_from_file_location(f"cap250_{archive.stem}", directory / "submission.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module, directory


def load(archive, channels, device):
    if archive.name == "stock.zip" and digest(archive) != STOCK_SHA:
        raise ValueError("Signed stock ZIP has changed")
    with open_package(archive) as (module, directory):
        if archive.name != "stock.zip" and module.LANE_B_ARM != archive.stem:
            raise ValueError(f"Wrong arm in {archive}")
        return module.Solver().load_model(dict(submission_dir=directory, device=device, ch_names=channels))


def fresh_tail(archive):
    with open_package(archive) as (module, directory):
        with np.load(directory / "weights.npz", allow_pickle=False) as z:
            return module.to_windows, module.CountFilter(z), module.CausalPers(z)


def check_stage(stage):
    log = stage.log
    if log["err"]:
        raise RuntimeError(f"{type(stage).__name__} encountered {log['err']} errors: {log['first']}")


def tail_meta(count, csp, classes):
    check_stage(count)
    check_stage(csp)
    return dict(count_age=count.latch, count_windows=count.t, csp_refits=csp.refits, csp_fusions=csp.fused,
                csp_single_class=csp.nfail, csp_missing_class_rows=sum(k == 2 for k in classes))


def from_trace(t):
    return any(g is not None and g[2] > 0 for g in (t["contrast"], t["identity"]))


def stream(template, windows, *, force=None):
    model = fresh_model(template)
    if force is not None:
        model.route.contrast.step = lambda *_: None
        model.route.identity.step = lambda *_: (0, .28, .4) if force(model.log6["n"]) else None
    count, csp = model.cf, model.cp
    features, terminal, raw, labels, live, fused, warm, classes = ([] for _ in range(8))
    original = csp.step

    def capture(x, p, o, recognised):
        q = original(x, p, o, recognised)
        features.append(x.copy())
        terminal.append(q.copy())
        classes.append(len(csp.model[1].classes_) if csp.model is not None else 0)
        return q

    csp.step = capture
    start = perf_counter()
    try:
        for x in windows:
            y = model.predict(x[None])
            if y.dtype != torch.int64 or y.shape != (1,):
                raise ValueError("Expected one int64 decision")
            labels.append(int(y[0]))
            raw.append(model.trace["out"].copy())
            live.append(from_trace(model.trace))
            fused.append(csp.fused)
            warm.append(bool(model.trace["v6"].get("p_ft")))
        if windows.is_cuda:
            torch.cuda.synchronize(windows.device)
    finally:
        csp.step = original
    wall = perf_counter() - start
    for stage in (model.route, model.routeB):
        check_stage(stage)
    if model.log6["err"]:
        raise RuntimeError(model.log6["first"])
    out = np.stack(terminal)
    np.testing.assert_array_equal(out.argmax(1), labels)
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite terminal scores")
    meta = tail_meta(count, csp, classes)
    meta.update(wall_s=wall, ft_refits=model.ft.refits, ft_steps=model.ft.steps, ft_s=model.ft.ft_sec,
                capped=model.ft.capped, cap=model.ft.cap, rec=model.log6["rec"], ft_inputs=len(model.ft.lab),
                recognition_returns=int(np.count_nonzero(np.asarray(live[:-1]) & ~np.asarray(live[1:]))))
    return dict(raw=np.stack(raw), terminal=out, labels=np.array(labels, np.int8), live=np.array(live), x=features,
                fused=np.array(fused), warm=np.array(warm), meta=meta)


def replay_tail(parts, features, raw, live, reversible):
    _, count, csp = parts
    count.reset_state()
    csp.reset_state()
    out, classes, latched = [], [], False
    for x, p, routed in zip(features, raw, live, strict=True):
        latched |= routed
        out.append(csp.step(x, p, count.step(x, p), routed if reversible else latched))
        classes.append(len(csp.model[1].classes_) if csp.model is not None else 0)
    scores = np.stack(out)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite replayed terminal scores")
    return scores, tail_meta(count, csp, classes)


def bitwise(a, b, what):
    if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
        size = float(np.max(np.abs(a.astype(float) - b))) if a.shape == b.shape else None
        raise ValueError(f"Bitwise guard failed for {what}; maximum difference {size}")


def assert_equal(a, b):
    for field in ("labels", "raw", "terminal", "live", "fused", "warm"):
        bitwise(a[field], b[field], field)
