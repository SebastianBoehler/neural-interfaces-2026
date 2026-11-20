"""Lane D3 returning instrument (copy of laneR4/ret_worker.py, same Lane A cases, causal EA, native zero control): arms = final stack (hardcb, first) and D0z/D3f/D3g
built by laneD3/build_drift.py on the fold package with the person's d3 moments (laneD3/add_d3.py), packages /work/pkg_ret/WHO_ARM.

Original doc: one (order, visit) of the conditional returning replay.

python stack_worker.py ORDER P04|2 [--smoke]. Copy of the Lane A dose_worker (same fold bundle, causal EA banks, live refits,
count filter, cp) with the packaged members instead of the doses. Arms: zero (unmodified fold package without EA substitution,
must equal the authenticated native-zero cell bitwise), stock (unmodified, causal EA: the control), hard / cb / hardcb
(`/work/pkg_stack/WHO_ARM`, built by build_fold.py: the same StackEnsemble code as the final package; hard actors exclude the
whole person, CBraMod fits exclude the P04/P10 visits 2-3). Checks that fail loudly: gates, shadow, v5 tapes and recognition
age identical in all arms; every member arm moves the scores; v6 error counters zero. No task label is read here."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import inputs as I
from dose_worker import SAME as _SAME, native_worker, prepared, run  # Lane A helpers (same release import path)
from lane_inputs import ORDERS, load

ARMS = tuple(os.environ.get("RET_ARMS", "hardcb+D3f+D3g").split("+"))  # final stack (first) and keep-adapting variants
SAME = tuple(k for k in _SAME if k not in ("mix", "outer"))  # v5 chain tapes; the v6 mix6 legitimately differs


def run_d3(native, template, case, n, arm):
    held, fresh = [], native.fresh_model  # the release's session() builds its own copy: capture it to read the counters
    native.fresh_model = lambda m: held.append(fresh(m)) or held[-1]
    try:
        tape, _, stats = native.session(template, "causal" if arm != "zero" else "native_zero", case, n)
    finally:
        native.fresh_model = fresh
    assert len(held) == 1
    d3 = getattr(held[0], "d3", None)
    return tape, dict(stats, routed=int(held[0].routeB.log["routed"]), d3=dict(d3.stats) if d3 is not None else None)


def main(order, visit, smoke):
    assert ARMS[0] == "hardcb"
    if torch.__version__ != "2.14.1+cu130" or torch.cuda.is_initialized():
        raise ValueError("requires the original clean SIF CPU runtime")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    who, day = visit.split("|")
    path, fit, names, weights, channels, cases = load(who, order)
    native = native_worker()
    orig = prepared(path, channels, who, f"orig_{who}", native)
    case = cases[day]
    n = int(os.environ.get("RET_N") or 0) or (31 if smoke else len(case["rows"]))
    tapes, stats = {}, {}
    tapes["zero"], stats["zero"] = run(native, orig, case, n, "zero")
    np.testing.assert_array_equal(tapes["zero"]["scores"], case["score"][:n])
    np.testing.assert_array_equal(tapes["zero"]["labels"], case["labels"][:n])
    tapes["stock"], stats["stock"] = run(native, orig, case, n, "stock")
    times = {}
    for arm in ARMS:
        model = prepared(Path("/work/pkg_ret") / f"{who}_{arm}", channels, who, f"{arm}_{who}", native)
        assert type(model).__name__ == "StackEnsemble", "member package did not load the stack class"
        assert arm == "hardcb" or model.d3.mode == {"D0z": 0, "D3f": 1, "D3g": 2}[arm], f"{arm}: wrong drift mode"
        t0 = time.perf_counter()
        tapes[arm], stats[arm] = run_d3(native, model, case, n, arm)
        times[arm] = time.perf_counter() - t0
    base, rec = tapes["stock"], stats["stock"]["recognition"]
    hc = tapes["hardcb"]
    for arm in ARMS:
        t, s = tapes[arm], stats[arm]
        for key in SAME:
            np.testing.assert_array_equal(t[key], base[key], err_msg=f"{arm}: {key} depends on the members")
        if s["recognition"] != rec:
            raise ValueError(f"{arm}: recognition differs")
        s["seconds"] = times[arm]
        s["max_score_change"] = float(np.abs(t["scores"] - base["scores"]).max())
        if arm == "hardcb":
            if s["max_score_change"] <= 1e-9:
                raise ValueError("hardcb: member arm did not move the scores: silent no-op")
            continue
        s["change_vs_hardcb"] = float(np.abs(t["scores"] - hc["scores"]).max())
        if arm == "D0z":
            for key in t:
                np.testing.assert_array_equal(t[key], hc[key], err_msg="D0z is not the final stack")
            continue
        d3 = s["d3"]
        first = np.flatnonzero(np.abs(t["scores"] - hc["scores"]).max(1) > 0) if t["scores"].ndim > 1 else np.flatnonzero(t["scores"] != hc["scores"])
        s["first_change"] = int(first[0]) if len(first) else -1
        if rec < 0 and s["change_vs_hardcb"] != 0.0:
            raise ValueError(f"{arm}: unrecognised session changed")
        if len(first) and (rec < 0 or first[0] <= rec or d3["active"] <= 0):
            raise ValueError(f"{arm}: first change at {first[0]}, recognition {rec}, active {d3['active']}: not post-latch")
        if rec >= 0 and d3["routed"] != s["routed"]:
            raise ValueError(f"{arm}: drift saw {d3['routed']} routed windows, RouteB {s['routed']}")
        if d3["nan_row"] and rec >= 0:
            raise ValueError(f"{arm}: {d3['nan_row']} routed windows had no moment row (wrong index?)")
        if rec >= 0 and rec + 45 <= n and d3["active"] <= 0 and arm == "D3f":
            raise ValueError(f"{arm}: recognised at {rec}, {n} windows, never active: silent no-op")
    payload = dict(rows=case["rows"][:n])
    for arm, t in tapes.items():
        payload.update({f"{arm}|{k}": v for k, v in t.items() if k in ("scores", "labels", "contrast", "identity")})
    out = Path(f"{order}_{who}_{day}" + ("_smoke" if smoke else ""))
    np.savez_compressed(out.with_suffix(".npz"), **payload)
    receipt = dict(order=order, visit=visit, n=n, recognition=rec, stats=stats, scope="conditional returning, no gold")
    out.with_suffix(".json").write_text(json.dumps(receipt, sort_keys=True, indent=1, default=float))
    print(json.dumps({a: (s["recognition"], round(s.get("change_vs_hardcb", s.get("max_score_change", 0)), 4), s.get("d3")) for a, s in stats.items()}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("order", choices=ORDERS)
    ap.add_argument("visit", choices=I.BASELINES)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    main(a.order, a.visit, a.smoke)
