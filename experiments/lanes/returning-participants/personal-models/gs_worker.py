"""Lane G gS vs stock on one (order, visit) of Lane A's conditional returning replay (same instrument as dose_worker.py).

python gs_worker.py ORDER P04|2 [--smoke]. Arms: zero (must equal the authenticated native zero), stock, ctl (stock row rewritten
through the same package path; must be bitwise stock), gS (own_s row of the held person from a visit-1-only g4 fit). Checks that fail
loudly: ctl == stock on every tape; gates/recognition/mix tapes identical across arms; gS routed > 0 must move the scores.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import inputs as I
from dose_worker import native_worker, prepared, run
from lane_inputs import ORDERS, load

SAME = ("u", "core", "shadow", "mix", "contrast", "identity", "latch")


def main(order, visit, smoke):
    if torch.__version__ != "2.14.1+cu130" or torch.cuda.is_initialized():
        raise ValueError("requires the original clean SIF CPU runtime")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    who, day = visit.split("|")
    path, fit, names, weights, channels, cases = load(who, order)
    native = native_worker()
    models = {a: prepared(p, channels, who, f"{a}_{who}", native)
              for a, p in (("stock", path), ("ctl", Path("/work/pkg_gs") / f"{who}_ctl"), ("gS", Path("/work/pkg_gs") / f"{who}_s23"))}
    case = cases[day]
    n = 31 if smoke else len(case["rows"])
    zt, _ = run(native, prepared(path, channels, who, f"zero_{who}", native), case, n, "zero")
    np.testing.assert_array_equal(zt["scores"], case["score"][:n])
    tapes, stats = {}, {}
    for a, m in models.items():
        tapes[a], stats[a] = run(native, m, case, n, "causal")
    base = tapes["stock"]
    for a in ("ctl", "gS"):
        for k in SAME:
            np.testing.assert_array_equal(tapes[a][k], base[k], err_msg=f"{a}: {k}")
        if stats[a]["recognition"] != stats["stock"]["recognition"]:
            raise ValueError(f"{a}: recognition differs")
    if not all(np.array_equal(tapes["ctl"][k], base[k]) for k in tapes["ctl"]):
        raise ValueError("ctl is not bitwise stock")
    routed = stats["stock"]["routed"]
    moved = float(np.abs(tapes["gS"]["scores"] - base["scores"]).max())
    changed = int((tapes["gS"]["labels"] != base["labels"]).sum())
    if routed > 0 and moved <= 1e-9:
        raise ValueError("gS routed windows but scores unchanged: silent no-op")
    if routed == 0 and moved != 0.0:
        raise ValueError("unrouted session changed")
    out = Path(f"{order}_{who}_{day}" + ("_smoke" if smoke else ""))
    payload = dict(rows=case["rows"][:n])
    for a, t in tapes.items():
        payload.update({f"{a}|{k}": v for k, v in t.items() if k in ("scores", "labels")})
    np.savez_compressed(out.with_suffix(".npz"), **payload)
    out.with_suffix(".json").write_text(json.dumps(dict(order=order, visit=visit, n=n, routed=routed, max_score_change=moved,
                                                         windows_decision_changed=changed, recognition=stats["stock"]["recognition"]),
                                                    indent=1, default=float))
    print(order, visit, "routed", routed, "max_score_change", moved, "decisions changed", changed, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("order", choices=ORDERS)
    ap.add_argument("visit", choices=I.BASELINES)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    main(a.order, a.visit, a.smoke)
