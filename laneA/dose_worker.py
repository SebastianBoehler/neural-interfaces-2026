"""Lane A: one (order, visit) of the conditional returning replay, stock control plus every returning dose.

python dose_worker.py ORDER P04|2 [--smoke]. Arms (all causal-EA, real fold package, live refits, count filter, cp):
  zero   unmodified fold package without EA substitution; must equal the authenticated native-zero cell, bit for bit
  stock  unmodified fold package with the order's causal EA means (the control every gain is measured against)
  k1 k2 k4 k8 k16 hard   the dosed copy (submissions/lean_v66_dose), dose 1/16 * k; hard = head alone after the ramp
Checks that fail loudly: k1 == stock bitwise on every tape; gates, shadow, EA-substituted mix, latch and recognition
age identical in all arms; a dose that never routed is bitwise stock; a dose that routed must move the scores and have
applied exactly one routed_mix per routed window. No task label is read here.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

import inputs as I
from lane_inputs import ORDERS, load
from record import load_model

ARMS = {"k1": (1.0, False), "k2": (2.0, False), "k4": (4.0, False), "k8": (8.0, False), "k16": (16.0, False),
        "hard": (1.0, True)}
SAME = ("u", "core", "shadow", "mix", "outer", "out_v5", "contrast", "identity", "latch")  # dose-independent tapes


def native_worker():
    spec = importlib.util.spec_from_file_location("native_inner_worker", Path(I.__file__).with_name("worker.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepared(path, channels, who, tag, native):
    model = load_model(path, channels, tag, "cpu")
    if Path(type(model).step.__globals__["__file__"]).resolve() != (path / "submission.py").resolve():
        raise ValueError("wrong package module")
    I.native_ft_proof(model, who)
    model.js_bank = None
    with np.load(I.BASE / "neural-interfaces-2026/data/proteus_windows.npz", allow_pickle=False) as data:
        own = (data["subject"] == f"Dreyer2026Proteus/{who}") & (data["session"] == "1")
        prewarm = data["X"][own][:17]
    for x in prewarm:  # the original §321 runtime warm-up, then independent copied sessions
        model.predict(torch.from_numpy(x[None]))
    model.reset_state()
    native.zero(model)
    return model


def run(native, template, case, n, arm):
    held, fresh = [], native.fresh_model  # the release's session() builds its own copy: capture it to read the counters
    native.fresh_model = lambda m: held.append(fresh(m)) or held[-1]
    try:
        tape, _, stats = native.session(template, "causal" if arm != "zero" else "native_zero", case, n)
    finally:
        native.fresh_model = fresh
    assert len(held) == 1
    route = held[0].routeB
    stats = dict(stats, dosed=int(getattr(route, "dosed", 0)), routed=int(route.log["routed"]),
                 dose=getattr(route, "dose", None), hard=getattr(route, "hard", False))
    return tape, stats


def main(order, visit, smoke):
    if torch.__version__ != "2.14.1+cu130" or torch.cuda.is_initialized():
        raise ValueError("requires the original clean SIF CPU runtime")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    who, day = visit.split("|")
    path, fit, names, weights, channels, cases = load(who, order)
    dose_path = Path("/work/pkg") / f"{who}_s23"
    native = native_worker()
    orig = prepared(path, channels, who, f"orig_{who}", native)
    dosed = prepared(dose_path, channels, who, f"dose_{who}", native)
    assert type(dosed.routeB).__dict__.get("configure") and not hasattr(orig.routeB, "configure")
    case = cases[day]
    n = 31 if smoke else len(case["rows"])
    tapes, stats = {}, {}
    tapes["zero"], stats["zero"] = run(native, orig, case, n, "zero")
    np.testing.assert_array_equal(tapes["zero"]["scores"], case["score"][:n])  # authenticated native zero of this order
    np.testing.assert_array_equal(tapes["zero"]["labels"], case["labels"][:n])
    tapes["stock"], stats["stock"] = run(native, orig, case, n, "stock")
    for arm, (k, hard) in ARMS.items():
        model = copy.deepcopy(dosed)
        model.routeB.configure(k, hard)
        tapes[arm], stats[arm] = run(native, model, case, n, arm)
    base, rec = tapes["stock"], stats["stock"]["recognition"]
    for arm in ARMS:
        t, s = tapes[arm], stats[arm]
        for key in SAME:
            np.testing.assert_array_equal(t[key], base[key], err_msg=f"{arm}: {key} depends on the dose")
        if s["recognition"] != rec or s["refit_ages"] != stats["stock"]["refit_ages"]:
            raise ValueError(f"{arm}: recognition or refit differs")
        if s["dosed"] != s["routed"] or s["routed"] != int((np.maximum(t["contrast"][:, 2], t["identity"][:, 2]) > 0).sum()):
            raise ValueError(f"{arm}: dosed {s['dosed']} routed {s['routed']} windows do not match the ramp tape")
        moved = float(np.abs(t["scores"] - base["scores"]).max())
        s["max_score_change"] = moved
        if arm == "k1" and not all(np.array_equal(t[k_], base[k_]) for k_ in t):
            raise ValueError("dose 1/16 is not bitwise stock")
        if arm != "k1" and s["routed"] > 0 and moved <= 1e-9:
            raise ValueError(f"{arm}: routed windows but no change: silent no-op")
        if s["routed"] == 0 and moved != 0.0:
            raise ValueError(f"{arm}: unrouted session changed")
    payload = dict(rows=case["rows"][:n])
    for arm, t in tapes.items():
        payload.update({f"{arm}|{k}": v for k, v in t.items() if k in ("scores", "labels", "contrast", "identity")})
    out = Path(f"{order}_{who}_{day}" + ("_smoke" if smoke else ""))
    np.savez_compressed(out.with_suffix(".npz"), **payload)
    receipt = dict(order=order, visit=visit, n=n, recognition=rec, stats=stats, scope="conditional returning, no gold")
    out.with_suffix(".json").write_text(json.dumps(receipt, sort_keys=True, indent=1, default=float))
    print(json.dumps({a: (s["recognition"], s["routed"], s["dosed"], round(s["max_score_change"], 4) if "max_score_change" in s else None)
                      for a, s in stats.items()}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("order", choices=ORDERS)
    ap.add_argument("visit", choices=I.BASELINES)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    main(a.order, a.visit, a.smoke)
