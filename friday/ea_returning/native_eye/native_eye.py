"""Substitute three fold-fitted Eye traces in the actual §321 RouteB worker."""

import argparse
import json
from pathlib import Path
import runpy

import numpy as np
import torch

import benchmark_utils.base_solver
from eye_scope import DATA, FIT, FOLDS, OCUNET, ORIGINAL_TRACES, VISITS, sha
from inputs import BASELINES, load, native_ft_proof
from record import load_model
from reference_eye import gallery, original_traces, reference
from returning_pair_guard import zero
from verify_fit import artifacts
from worker import same_prefix, session


FIT_SOURCE = "a2566e46004374d08238f4abf2fb9846486910fd2ec173c66cbc4e1a1f6c65dd"
N_SMOKE, N_PREFIX = 31, 17
UNCHANGED = ("mix", "shadow", "u", "core", "contrast", "identity", "latch", "out_v5", "unpatched_ea", "outer")


class ObserveEye:
    def __init__(self, model):
        self.model = model
        if type(model.eye).__name__ != "EyeNet" or len(model.eye.nets) != 3 or model.v6r:
            raise ValueError("native Eye owner or v5 refit path differs")
        self.owner = model.eye.traces
        model.eye.traces = self.trace
        self.traces, self.extra, self.rest = [], [], []

    def trace(self, x):
        h = self.owner(x)
        self.traces.append(h.copy())
        return h

    def inspect(self, trace, t):
        m, v = self.model, trace["v6"]
        if len(self.traces) != t + 1 or m.eye.n != t + 1 or len(m.routeB.tail.em.A) != t + 1:
            raise ValueError(f"actual native Eye or consuming EM route did not run at {t}")
        if v["extra"].shape != (3,) or not np.isfinite(v["extra"]).all():
            raise ValueError("actual native Eye evidence is invalid")
        np.testing.assert_array_equal(m.routeB.tail.em.A[-1][-3:], v["extra"])
        self.extra.append(v["extra"].copy())
        self.rest.append(v["rest"].copy())

    def arrays(self):
        return {key: np.asarray(getattr(self, key)) for key in ("traces", "extra", "rest")}


def call(model, arm, case, length):
    probes = []

    def instrument(current):
        probe = ObserveEye(current)
        probes.append(probe)
        return probe.inspect

    run = session(model, arm, case, length, instrument)
    return run, probes[0].arrays()


def independent_traces(case, n, channels, actual):
    x = case["x"][:n]
    vector = runpy.run_path(str(OCUNET))["traces"](
            dict(X=x, ch_names=channels, stream=np.zeros(n, dtype=int)))
    if vector.shape != (n, 7, 480) or actual.shape != vector.shape or not np.isfinite(actual).all():
        raise ValueError("owning seven-trace Eye geometry differs")
    delta = float(np.abs(vector - actual).max())
    if delta > 2e-5:
        raise ValueError(f"native ocular trace differs from its owning reference: {delta}")
    return delta


def check_prefix(model, case, other, reference):
    call(model, "causal", other, 9)
    prefix, arrays = call(model, "causal", case, N_PREFIX)
    same_prefix(reference[0], prefix, N_PREFIX)
    for key in arrays:
        np.testing.assert_array_equal(reference[1][key][:N_PREFIX], arrays[key])


def prepare(who):
    root, fit, names, weights, channels, cases = load(who)
    if fit["fit_sha"] != FOLDS[who][4] or len(fit["held"]) != FOLDS[who][2]:
        raise ValueError("wrong source and session scope")
    original_traces(who, root)
    model = load_model(root, channels, f"source_eye_{who}", "cpu")
    if Path(type(model).step.__globals__["__file__"]).resolve() != (root / "submission.py").resolve():
        raise ValueError("wrong actual §321 owner")
    np.testing.assert_array_equal(model.w, weights)
    native_ft_proof(model, who)
    model.js_bank = None
    calibration = gallery(model, who, fit["fit_sha"])
    with np.load(DATA, allow_pickle=False) as d:
        own = np.flatnonzero((d["subject"] == f"Dreyer2026Proteus/{who}") & (d["session"] == "1"))[:17]
        for x in d["X"][own]:
            model.predict(torch.from_numpy(x[None]))
    model.reset_state()
    zero(model)
    return model, channels, cases, calibration


def replay(slot, smoke):
    if (torch.__version__ != "2.14.1+cu130" or torch.cuda.is_initialized()
            or "neural-interfaces-2026/.venv/" in Path("/proc/self/maps").read_text()):
        raise ValueError("expected the pinned clean native CPU worker")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    who, day, length = VISITS[slot]
    n = N_SMOKE if smoke else length
    traces, fit_smokes = artifacts(who, FIT_SOURCE)
    if any(x["trace"] in ORIGINAL_TRACES for x in traces):
        raise ValueError("still using a production Eye trace")
    model, channels, cases, calibration = prepare(who)
    case, other = cases[day], cases["3" if day == "2" else "2"]
    archive, provenance = reference(slot)
    native_zero, _ = call(model, "native_zero", case, n)
    before = call(model, "causal", case, n)
    check_prefix(model, case, other, before)
    with np.load(archive, allow_pickle=False) as z:
        np.testing.assert_array_equal(z["rows"], case["rows"])
        for arm, result in (("native_zero", native_zero), ("causal", before[0])):
            for key in ("scores", "labels", "mix", "shadow", "contrast", "identity", "latch"):
                np.testing.assert_array_equal(result[0][key], z[f"{arm}|{key}"][:n])
    files = [Path(x["path"]).name for x in traces]
    model.eye = type(model.eye)(dict(v6_eye_files=files), "cpu", FIT / "fits" / who)
    model.reset_state()
    zero(model)
    after = call(model, "causal", case, n)
    check_prefix(model, case, other, after)
    old, new = before[0][0], after[0][0]
    for key in UNCHANGED:
        np.testing.assert_array_equal(new[key], old[key])
    for key in ("traces", "rest"):
        np.testing.assert_array_equal(before[1][key], after[1][key])
    activity = before[0][2]
    expected_ages = [20] if slot == 2 else []
    if (activity != after[0][2] or activity["recognition"] != BASELINES[f"{who}|{day}"]
            or activity["refit_ages"] != expected_ages or activity["steps"] != 20 * len(expected_ages)):
        raise ValueError("the underlying real warm or recognition path changed")
    same_prefix(before[0], (old, after[0][1], activity), n)
    ocular_error = independent_traces(case, n, channels, after[1]["traces"])
    evidence = float(np.abs(after[1]["extra"] - before[1]["extra"]).max())
    terminal = float(np.abs(new["scores"] - old["scores"]).max())
    if evidence <= .15 or terminal <= 1e-4:
        raise ValueError(f"new Eye evidence did not reach the real terminal output {(evidence, terminal)}")
    path = Path("eye_replay.npz")
    with path.open("xb") as f:
        np.savez_compressed(f, rows=case["rows"][:n], stock_labels=old["labels"], stock_scores=old["scores"],
                            fresh_labels=new["labels"], fresh_scores=new["scores"],
                            stock_eye=before[1]["extra"], fresh_eye=after[1]["extra"])
    path.chmod(0o444)
    proof = dict(source=sha(Path(__file__).parent / "source.sha256"), order="full0", person=who, visit=day,
                 slot=slot, rows=n, smoke=smoke, held_labels_opened=False, accuracy_computed=False,
                 source_fit=FIT_SOURCE, traces=traces, fit_smokes=fit_smokes, fixed_old_oof_gallery=calibration,
                 native_ref=provenance, exact_native_zero=True, exact_native_control=True,
                 eye_routeb_evidence_exact=True, rest_inner_and_outer_mix_exact=True, warm_exact=True,
                 prefix_rows=N_PREFIX, prefix_reset_exact=True, original_eye=list(ORIGINAL_TRACES),
                 activity=activity, native_ocular_trace_max_abs=ocular_error, max_eye_log_delta=evidence,
                 max_terminal_delta=terminal, changed=int(np.count_nonzero(new["labels"] != old["labels"])),
                 output=sha(path))
    receipt = path.with_suffix(".json")
    with receipt.open("x") as f:
        json.dump(proof, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    receipt.chmod(0o444)
    print("PASS authenticated native Eye replay", who, day, n, sha(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slot", type=int, choices=range(len(VISITS)))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    replay(args.slot, args.smoke)
