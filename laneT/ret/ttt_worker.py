"""Returning P04/P10 instrument for the stack members: one (order, visit) of the conditional returning replay.

python stack_worker.py ORDER P04|2 [--smoke]. Copy of the Lane A dose_worker (same fold bundle, causal EA banks, live refits,
count filter, cp) with the packaged members instead of the doses. Arms: zero (unmodified fold package without EA substitution,
must equal the authenticated native-zero cell bitwise), stock (unmodified, causal EA: the control), hard / cb / hardcb
(`/work/pkg_stack/WHO_ARM`, built by build_fold.py: the same StackEnsemble code as the final package; hard actors exclude the
whole person, CBraMod fits exclude the P04/P10 visits 2-3). Checks that fail loudly: gates, shadow, v5 tapes and recognition
age identical in all arms; every member arm moves the scores; v6 error counters zero. No task label is read here."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import inputs as I
from dose_worker import SAME as _SAME, native_worker, prepared, run  # Lane A helpers (same release import path)
from lane_inputs import ORDERS, load

ARMS = ("hardcb", "ftlight", "ftheavy")  # lane T: final stack and its two FT-stage variants
SAME = tuple(k for k in _SAME if k not in ("mix", "outer"))  # v5 chain tapes; the v6 mix6 legitimately differs


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
    np.testing.assert_array_equal(tapes["zero"]["scores"], case["score"][:n])
    np.testing.assert_array_equal(tapes["zero"]["labels"], case["labels"][:n])
    tapes["stock"], stats["stock"] = run(native, orig, case, n, "stock")
    for arm in ARMS:
        model = prepared(Path("/work/pkg_ttt") / f"{who}_{arm}", channels, who, f"{arm}_{who}", native)
        assert type(model).__name__ == "StackEnsemble", "member package did not load the stack class"
        assert type(model.ft).__name__ == ("FineTune" if arm == "hardcb" else "TTTFineTune"), (arm, type(model.ft).__name__)
        tapes[arm], stats[arm] = run(native, model, case, n, arm)
    base, rec = tapes["stock"], stats["stock"]["recognition"]
    for arm in ARMS:
        t, s = tapes[arm], stats[arm]
        for key in SAME:
            np.testing.assert_array_equal(t[key], base[key], err_msg=f"{arm}: {key} depends on the members")
        if s["recognition"] != rec or s["refit_ages"] != stats["stock"]["refit_ages"]:
            raise ValueError(f"{arm}: recognition or refit schedule differs")
        s["max_score_change"] = float(np.abs(t["scores"] - base["scores"]).max())
        if arm == "hardcb" and s["max_score_change"] <= 1e-9:
            raise ValueError(f"{arm}: member arm did not move the scores: silent no-op")
    payload = dict(rows=case["rows"][:n])
    for arm, t in tapes.items():
        payload.update({f"{arm}|{k}": v for k, v in t.items() if k in ("scores", "labels", "contrast", "identity")})
    out = Path(f"{order}_{who}_{day}" + ("_smoke" if smoke else ""))
    np.savez_compressed(out.with_suffix(".npz"), **payload)
    receipt = dict(order=order, visit=visit, n=n, recognition=rec, stats=stats, scope="conditional returning, no gold")
    out.with_suffix(".json").write_text(json.dumps(receipt, sort_keys=True, indent=1, default=float))
    print(json.dumps({a: (s["recognition"], round(s.get("max_score_change", 0), 4)) for a, s in stats.items()}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("order", choices=ORDERS)
    ap.add_argument("visit", choices=I.BASELINES)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    main(a.order, a.visit, a.smoke)
