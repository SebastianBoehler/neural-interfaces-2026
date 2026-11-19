"""Whole-system calibrated Eye replay with exact production and source-only controls."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import benchmark_utils.base_solver
from eye_cal_scope import FIT_SOURCE, MASS
from eye_embedding import load_networks
from eye_scope import BUNDLE, VISITS, sha
from head_inputs import HEAD_SOURCE, load
from native_eye import N_PREFIX, N_SMOKE, UNCHANGED, call as original_call, independent_traces, prepare
from reference_eye import reference
from returning_pair_guard import zero
from route_eye import RouteEye
from worker import same_prefix, session


ARMS = ("production", "source", "calibrated")


def call(model, case, n, heads, nets, names, active):
    probes = []

    def observer(current):
        p = RouteEye(current, heads if active else {}, nets, names)
        probes.append(p)
        return p.inspect

    result = session(model, "causal", case, n, observer)
    return result, probes[0].arrays(), probes[0].peak


def causal_prefix(model, case, other, reference, heads, nets, names, active):
    call(model, other, 9, heads, nets, names, active)
    run, arrays, _ = call(model, case, N_PREFIX, heads, nets, names, active)
    same_prefix(reference[0], run, N_PREFIX)
    for k in arrays:
        np.testing.assert_array_equal(reference[1][k][:N_PREFIX], arrays[k])


def identical_stage(old, other, n):
    left, right = old[0][0], other[0][0]
    for k in UNCHANGED:
        np.testing.assert_array_equal(right[k], left[k])
    for key in ("rest", "traces"):
        np.testing.assert_array_equal(old[1][key], other[1][key])
    if old[0][2] != other[0][2]:
        raise ValueError("native optimizer or recognition changed")
    same_prefix(old[0], (left, other[0][1], other[0][2]), n)


def compare_evidence(source, candidate):
    a, b = source[1], candidate[1]
    ramp = b["ramp"]
    active = ramp > 0
    if not active.any() or not (0 < ramp[active].min() <= ramp[active].max() <= 1):
        raise ValueError("recognition-gated Eye head never ran")
    np.testing.assert_array_equal(a["raw"], b["raw"])
    np.testing.assert_array_equal(a["raw"], a["extra"])
    np.testing.assert_array_equal(a["extra"][~active], b["extra"][~active])
    first = int(np.flatnonzero(active)[0])
    np.testing.assert_array_equal(source[0][0]["scores"][:first], candidate[0][0]["scores"][:first])
    p = np.exp(a["raw"][active])
    p /= p.sum(1, keepdims=True)
    mass = MASS * ramp[active, None]
    expected = np.log(np.clip((1 - mass) * p + mass * b["personal"][active], 1e-3, 1))
    error = float(np.abs(expected - b["extra"][active]).max())
    evidence = float(np.abs(a["extra"] - b["extra"]).max())
    terminal = float(np.abs(source[0][0]["scores"] - candidate[0][0]["scores"]).max())
    if error > 1e-14 or evidence <= .01 or terminal <= 1e-4:
        raise ValueError(f"calibrated evidence failed to reach the active RouteB terminal {(error, evidence, terminal)}")
    return dict(active_windows=int(active.sum()), first_active=first, identifiers=sorted(set(b["person"][active])),
                blend_max_abs=error, evidence_max_abs=evidence, terminal_max_abs=terminal)


def replay(slot, smoke):
    if (torch.__version__ != "2.14.1+cu130" or torch.cuda.is_initialized()
            or "neural-interfaces-2026/.venv/" in Path("/proc/self/maps").read_text()):
        raise ValueError("wrong actual native CPU runtime")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    who, day, length = VISITS[slot]
    n = N_SMOKE if smoke else length
    fits, heads, head_receipt = load()
    nets = load_networks(fits)
    model, channels, cases, gallery = prepare(who)
    case, other = cases[day], cases["3" if day == "2" else "2"]
    with np.load(BUNDLE / f"{who}_s23/base/weights.npz", allow_pickle=False) as z:
        names = [str(s).rsplit("/", 1)[-1] for s in z["route_names"]]
    if len(names) != len(set(names)) or not set(heads).issubset(names):
        raise ValueError("native route vocabulary does not contain both calibrated source people")
    native_zero, _ = original_call(model, "native_zero", case, n)
    old, extra = original_call(model, "causal", case, n)
    archive, anchor = reference(slot)
    with np.load(archive, allow_pickle=False) as z:
        np.testing.assert_array_equal(case["rows"], z["rows"])
        for arm, result in (("native_zero", native_zero), ("causal", old)):
            for key in ("scores", "labels", "mix", "shadow", "contrast", "identity", "latch"):
                np.testing.assert_array_equal(result[0][key], z[f"{arm}|{key}"][:n])
    path = Path(fits[0]["path"]).parent
    model.eye = type(model.eye)(dict(v6_eye_files=[Path(x["path"]).name for x in fits]), "cpu", path)
    model.reset_state()
    zero(model)
    source = call(model, case, n, heads, nets, names, False)
    calibrated = call(model, case, n, heads, nets, names, True)
    for active, arm in ((False, source), (True, calibrated)):
        causal_prefix(model, case, other, arm, heads, nets, names, active)
        identical_stage((old, extra), arm, n)
    checks = compare_evidence(source, calibrated)
    if checks["identifiers"] != [who] or calibrated[2] > 1e-5:
        raise ValueError("new Eye evidence used an unexpected native identity or trace")
    trace_error = independent_traces(case, n, channels, calibrated[1]["traces"])
    activity = old[2]
    expected = 28 if slot == 2 else 10
    if (activity["recognition"] != expected or activity["steps"] != (20 if slot == 2 else 0)
            or checks["first_active"] != expected):
        raise ValueError("registered native recognition or real warm activity did not match")
    output = Path("eye_cal.npz")
    tape = {k: r for k, r in zip(ARMS, (old[0], source[0][0], calibrated[0][0]), strict=True)}
    with output.open("xb") as f:
        np.savez_compressed(f, rows=case["rows"][:n],
                            **{f"{arm}_{k}": v[k] for arm, v in tape.items() for k in ("labels", "scores")})
    output.chmod(0o444)
    proof = dict(order="full0", slot=slot, person=who, visit=day, rows=n, smoke=smoke,
                 source=sha(Path(__file__).parent / "source.sha256"), held_labels_opened=False, scored=False,
                 source_fit=FIT_SOURCE, head_source=HEAD_SOURCE, head_receipt=head_receipt,
                 native_anchors_exact=True, independent_anchor=anchor,
                 old_oof_gallery=gallery, inner_warm_rest_outer_exact=True, prefix_reset_exact=True,
                 prefix_rows=N_PREFIX, original_signal_trace_max_abs=trace_error, ocular_net_max_abs=calibrated[2],
                 activation=checks, activity=activity, output=sha(output))
    seal = output.with_suffix(".json")
    with seal.open("x") as f:
        json.dump(proof, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    seal.chmod(0o444)
    print("PASS actual native Eye calibration", who, day, n, sha(seal), checks, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("slot", choices=range(4), type=int)
    cli.add_argument("--smoke", action="store_true")
    args = cli.parse_args()
    replay(args.slot, args.smoke)
