"""Lane I: one (order, visit) of the conditional returning replay (Lane A instrument): stock control plus the fold package with
its six raw SessionBN traces rewritten to prior count k20 (control, must equal stock bitwise), k10 and k5.

python bn_worker.py ORDER P04|2 [--smoke]. Arms (all causal-EA, real fold package, live refits incl. in-session fine-tuning, count filter, cp):
  zero   unmodified fold package without EA substitution; must equal the authenticated native-zero cell, bit for bit
  stock  unmodified fold package with the order's causal EA means (the control every gain is measured against)
  k20 k10 k5   same package, `sbn_fbcnet_raw_k20_s*.pt` constants patched by ret/patch_k.py (the fine-tuned member loads them too)
Fails loudly: k20 not bitwise stock on any tape; k10/k5 that leave every score unchanged. Gates equality is reported, not required.
No task label is read here.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

import inputs as I
from lane_inputs import ORDERS, load
from record import load_model

ARMS = ("k20", "k10", "k5")  # packages whose six raw SessionBN traces carry that baked prior count; k20 must equal stock bitwise


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
    stats = dict(stats, routed=int(route.log["routed"]))
    return tape, stats


def main(order, visit, smoke):
    if torch.__version__ != "2.14.1+cu130" or torch.cuda.is_initialized():
        raise ValueError("requires the original clean SIF CPU runtime")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    who, day = visit.split("|")
    path, fit, names, weights, channels, cases = load(who, order)
    native = native_worker()
    orig = prepared(path, channels, who, f"orig_{who}", native)
    case = cases[day]
    n = 31 if smoke else len(case["rows"])
    tapes, stats = {}, {}
    tapes["zero"], stats["zero"] = run(native, orig, case, n, "zero")
    np.testing.assert_array_equal(tapes["zero"]["scores"], case["score"][:n])  # authenticated native zero of this order
    np.testing.assert_array_equal(tapes["zero"]["labels"], case["labels"][:n])
    tapes["stock"], stats["stock"] = run(native, orig, case, n, "stock")
    for arm in ARMS:
        model = prepared(Path("/work/pkg") / f"{who}_{arm}", channels, who, f"{arm}_{who}", native)
        tapes[arm], stats[arm] = run(native, model, case, n, arm)
    base = tapes["stock"]
    if not all(np.array_equal(tapes["k20"][k_], base[k_]) for k_ in base):
        raise ValueError("patched k20 traces are not bitwise stock")
    for arm in ARMS[1:]:
        s = stats[arm]
        s["max_score_change"] = float(np.abs(tapes[arm]["scores"] - base["scores"]).max())
        if s["max_score_change"] <= 1e-9:
            raise ValueError(f"{arm}: no score change: silent no-op")
        s["gates_same_as_stock"] = all(np.array_equal(tapes[arm][k_], base[k_]) for k_ in ("contrast", "identity", "latch"))
    payload = dict(rows=case["rows"][:n])
    for arm, t in tapes.items():
        payload.update({f"{arm}|{k}": v for k, v in t.items() if k in ("scores", "labels", "contrast", "identity")})
    out = Path(f"{order}_{who}_{day}" + ("_smoke" if smoke else ""))
    np.savez_compressed(out.with_suffix(".npz"), **payload)
    receipt = dict(order=order, visit=visit, n=n, stats=stats, scope="conditional returning, no gold")
    out.with_suffix(".json").write_text(json.dumps(receipt, sort_keys=True, indent=1, default=float))
    print(json.dumps({a: (s.get("recognition"), s.get("routed"), round(s.get("max_score_change", 0.0), 4)) for a, s in stats.items()}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("order", choices=ORDERS)
    ap.add_argument("visit", choices=I.BASELINES)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    main(a.order, a.visit, a.smoke)
