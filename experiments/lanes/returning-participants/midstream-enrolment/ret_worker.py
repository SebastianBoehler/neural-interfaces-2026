"""Lane R4 returning instrument (copy of laneT/ret/ttt_worker.py): arms = final stack (hardcb, first) and keep-adapting variants built by build_keep.py.

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
from pathlib import Path

import numpy as np
import torch

import inputs as I
from dose_worker import SAME as _SAME, native_worker, prepared, run  # Lane A helpers (same release import path)
from lane_inputs import ORDERS, load

ARMS = tuple(os.environ.get("RET_ARMS", "hardcb+R1").split("+"))  # final stack (first) and keep-adapting variants
TTT_ARMS = ("R3", "R4")
SAME = tuple(k for k in _SAME if k not in ("mix", "outer"))  # v5 chain tapes; the v6 mix6 legitimately differs


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
    n = int(os.environ.get("RET_N") or 0) or (31 if smoke else len(case["rows"]))  # RET_N: longer smoke
    tapes, stats = {}, {}
    tapes["zero"], stats["zero"] = run(native, orig, case, n, "zero")
    np.testing.assert_array_equal(tapes["zero"]["scores"], case["score"][:n])
    np.testing.assert_array_equal(tapes["zero"]["labels"], case["labels"][:n])
    tapes["stock"], stats["stock"] = run(native, orig, case, n, "stock")
    for arm in ARMS:
        model = prepared(Path("/work/pkg_ret") / f"{who}_{arm}", channels, who, f"{arm}_{who}", native)
        assert type(model).__name__ == "StackEnsemble", "member package did not load the stack class"
        assert type(model.ft).__name__ == ("TTTFineTune" if arm in TTT_ARMS else "FineTune"), (arm, type(model.ft).__name__)
        tapes[arm], stats[arm] = run(native, model, case, n, arm)
    base, rec = tapes["stock"], stats["stock"]["recognition"]
    for arm in ARMS:
        t, s = tapes[arm], stats[arm]
        for key in SAME:
            np.testing.assert_array_equal(t[key], base[key], err_msg=f"{arm}: {key} depends on the members")
        if s["recognition"] != rec:
            raise ValueError(f"{arm}: recognition differs")
        s["max_score_change"] = float(np.abs(t["scores"] - base["scores"]).max())
        if arm == "hardcb":
            if s["max_score_change"] <= 1e-9:
                raise ValueError("hardcb: member arm did not move the scores: silent no-op")
            continue
        s["change_vs_hardcb"] = float(np.abs(t["scores"] - tapes["hardcb"]["scores"]).max())
        if arm == "R0z":
            for key in t:
                np.testing.assert_array_equal(t[key], tapes["hardcb"][key], err_msg="R0z is not the final stack")
            continue
        late = [a for a in s["refit_ages"] if rec >= 0 and a > rec]
        s["refits_after_recognition"] = len(late)
        if rec < 0 and s["change_vs_hardcb"] != 0.0:
            raise ValueError(f"{arm}: unrecognised session changed")
        if arm == "R5" and late:
            raise ValueError("R5: FT refit after recognition (FT must stay retired)")
        if arm == "R1a" and any(a != 100 for a in late):
            raise ValueError(f"R1a: refit ages after recognition {late}, expected only 100")
        if rec >= 0 and rec + 10 <= min(n - 1, 250) and not (arm == "R1a" and n <= 100) and not (arm == "R5" and n < 120):  # CSP fuses from window 90  # a refit grid point lies after recognition and before the cap
            if not late and arm != "R5" and not arm.startswith("R2q"):
                raise ValueError(f"{arm}: recognised at {rec} but no refit after it: silent no-op")
            if s["change_vs_hardcb"] <= 1e-9:
                raise ValueError(f"{arm}: keep arm did not move the scores vs hardcb: silent no-op")
        if arm not in ("R1", "R1a") and rec >= 0 and s["cp_stopped"] is not None:
            raise ValueError(f"{arm}: CSP still retired at {s['cp_stopped']}")
    payload = dict(rows=case["rows"][:n])
    for arm, t in tapes.items():
        payload.update({f"{arm}|{k}": v for k, v in t.items() if k in ("scores", "labels", "contrast", "identity")})
    out = Path(f"{order}_{who}_{day}" + ("_smoke" if smoke else ""))
    np.savez_compressed(out.with_suffix(".npz"), **payload)
    receipt = dict(order=order, visit=visit, n=n, recognition=rec, stats=stats, scope="conditional returning, no gold")
    out.with_suffix(".json").write_text(json.dumps(receipt, sort_keys=True, indent=1, default=float))
    print(json.dumps({a: (s["recognition"], round(s.get("change_vs_hardcb", s.get("max_score_change", 0)), 4)) for a, s in stats.items()}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("order", choices=ORDERS)
    ap.add_argument("visit", choices=I.BASELINES)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    main(a.order, a.visit, a.smoke)
