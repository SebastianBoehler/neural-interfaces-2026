"""Observe identical official stock sessions under one fixed process launch mode."""
import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import torch

import replay
from benchmark_utils.streaming import fresh_model
from mirror import Outer
from qualify import ARM_ORDER, STOCK, TOL, fixtures, measured, protect
from settings import sha, stamp
from stock_drift import capture, effects

IMAGE = "7fc0389c46ac6bec7c4395a9815e9ffee04d207a2a3700aeb3aef318bf79e2d0"
DIAGNOSIS = "ba3cc6f6d9bdf373ea065e76c293b2b7bee46b281ca729d71a12259e24e9a190"


def set_mode(mode):
    controlled = mode == "controlled"
    assert torch.__version__ == "2.14.1+cu130" and not torch.cuda.is_initialized()
    workspace = os.getenv("CUBLAS_WORKSPACE_CONFIG")
    assert workspace == (":4096:8" if controlled else None)
    if controlled:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    assert torch.are_deterministic_algorithms_enabled() == controlled
    assert torch.backends.cudnn.deterministic == controlled and not torch.backends.cudnn.benchmark
    torch.set_num_threads(1)
    device = torch.cuda.get_device_name()
    assert "RTX 2080 Ti" in device
    return dict(device=device, workspace=workspace, deterministic=controlled, cudnn_deterministic=controlled,
                benchmark=False, torch=torch.__version__)


def native_zero(template, result):
    independent = Outer(template)
    for x, u, mix, core, extra, gate, identity, recognized, _, _ in result["events"]:
        independent.step((x, u, mix, mix, core, extra, gate, identity, recognized), "stock")
    actual = np.stack((independent.raw, independent.count, independent.terminal), 1)
    np.testing.assert_array_equal(result["stages"], actual)
    assert (independent.cp.refits, independent.cp.fused) == (result["work"]["cp_refits"], result["work"]["cp_fused"])
    return True


def play(model, fixture, warm):
    x, _, _, expected, _, signed = fixture
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    record = measured(model, x)
    torch.cuda.synchronize()
    runtime = perf_counter() - start
    peak = torch.cuda.max_memory_reserved()
    data = record[1]
    np.testing.assert_array_equal(data["labels"], expected)
    assert data["work"] == signed["activity"] and native_zero(model, data)
    assert data["work"]["errors"] == data["work"]["ft_over"] == 0
    return record, dict(seconds=runtime, peak_cuda_reserved_bytes=peak, warm=warm, archive_exact=True)


def prefix(template, active, ref, x, peer):
    rows = 17
    checks = {}
    for tag, model in (("active_copy", fresh_model(active)), ("reused", active)):
        model.reset_state()
        model.predict(peer[0:1])
        model.reset_state()
        assert model.ft.nets is None and model.cf.t == model.cp.t == 0
        result = capture(model, x[:rows])
        np.testing.assert_array_equal(result["labels"], ref["labels"][:rows])
        assert result["work"]["windows"] == rows and native_zero(template, result)
        error = float(np.max(np.abs(result["stages"] - ref["stages"][:rows])))
        assert error <= TOL
        checks[tag] = error
    return checks


def repeat(model, case):
    a, runtime_a = play(model, case, True)
    b, runtime_b = play(model, case, True)
    drift = effects(a[1]["protected"], b[1]["protected"])
    protected = protect(a, b, None)
    maximum = max(v["max"] for v in drift.values())
    within = maximum <= TOL and all(protected[k] <= TOL for k in ("prior_weight_max", "gate_score_max"))
    return a, dict(work=a[1]["work"], native_mirrors_exact=True, labels_exact=True, per_stage=drift,
                   protected=protected, max_all=maximum, pass_independent=within,
                   runs=(runtime_a, runtime_b), input=stamp(case[0].cpu().numpy()), delivery=case[1].tolist())


def run(mode, smoke):
    flags = set_mode(mode)
    assert sha("/stock.zip") == STOCK and sha("/diagnosis.json") == DIAGNOSIS
    diag = json.loads(Path("/diagnosis.json").read_text())
    assert diag["stock"] == STOCK and not diag["held_labels_opened"]
    cases, channels, anchor = fixtures()
    g, p10 = (cases[k] for k in ARM_ORDER)
    assert diag["result"]["rows"] == g[1].tolist()
    with TemporaryDirectory() as tmp:
        with ZipFile("/stock.zip") as archive:
            archive.extractall(tmp)
        model = replay.load(Path(tmp), channels)
        assert type(model).__name__ == "Ensemble6"
        assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
        if smoke:
            _, observed, _, _ = measured(model, g[0][:31])
            np.testing.assert_array_equal(observed["labels"], g[3][:31])
            assert native_zero(model, observed)
            assert observed["work"]["ft_steps"] == 20 and observed["work"]["cp_refits"] == 1
            assert observed["work"]["recognized"] == -1 and observed["work"]["errors"] == 0
            path = Path(f"{mode}_smoke.json")
            with path.open("x") as handle:
                json.dump(dict(mode=mode, source=sha("source.sha256"), stock=STOCK, image=IMAGE,
                    flags=flags, held_labels_opened=False, passed=True, rows=31, work=observed["work"]),
                    handle, sort_keys=True, allow_nan=False)
                handle.write("\n")
            path.chmod(0o444)
            print("PASS active G technical smoke", mode, sha(path), flush=True)
            return
        witness = json.loads(Path(f"{mode}_smoke.json").read_text())
        assert witness["passed"] and witness["mode"] == mode and witness["source"] == sha("source.sha256")
        assert witness["stock"] == STOCK and witness["image"] == IMAGE and not witness["held_labels_opened"]
        _, cold = play(model, g, False)
        left, witness = repeat(model, g)
        _, known = repeat(model, p10)
        assert witness["work"]["recognized"] == -1 and witness["work"]["ft_steps"] == 200
        assert witness["work"]["cp_fused"] == 30
        assert known["work"]["recognized"] == 10 and known["work"]["ft_steps"] == 0
        reset = prefix(model, left[0], left[1], g[0], p10[0])
    path = Path(f"{mode}.json")
    with path.open("x") as handle:
        json.dump(dict(mode=mode, source=sha("source.sha256"), stock=STOCK, image=IMAGE, flags=flags,
                       diagnosis=DIAGNOSIS, cold_anchor=anchor, held_labels_opened=False,
                       cold=cold, G=witness, P10=known, reset=reset), handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
    path.chmod(0o444)
    print("PASS stock mode receipt", mode, sha(path), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("default", "controlled"))
    parser.add_argument("--smoke", action="store_true")
    a = parser.parse_args()
    run(a.mode, a.smoke)
