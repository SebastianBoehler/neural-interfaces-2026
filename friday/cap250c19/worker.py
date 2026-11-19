"""Re-run live count and CSP and seal all counterfactuals before scoring."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from bridge_orders import write_npz, write_receipt
from bridge_owner import sha
from complete_tail import count_filter, limit_session
from complete_zero import FEAT
from inputs import full0, within0
from tails import scopes

from capture import capture
from gate import decide

WORK = Path(__file__).resolve().parent
ARMS = ("stock", "no_ft")


def check_anchor(ctx, name, stage, rows, filtered):
    for k, actual in (("filtered", filtered[rows]), ("terminal", stage["scores"])):
        if k in ctx["matched"][name]:
            np.testing.assert_array_equal(actual, ctx["matched"][name][k][rows])
    assert np.isfinite(filtered[rows]).all()


def observe(ctx, sessions):
    count, order, final = (ctx[k] for k in ("count", "order", "final"))
    per_session = {}
    works, stages = {}, {}
    with scopes(count, sessions):
        rows = {s:count.C.order_rows(order, s, count.PROTO) for s in sessions}
        for arm in ARMS:
            start = perf_counter()
            raw = ctx["matched"][arm]["raw"]
            filtered, latch = count_filter(count, raw, order)
            count_seconds = perf_counter() - start
            stages[arm], works[arm] = {}, {}
            for s, ix in rows.items():
                start = perf_counter()
                tape, proof = capture(final, ctx["windows"], raw, filtered, ix)
                check_anchor(ctx, arm, tape, ix, filtered)
                works[arm][str(s)] = dict(**proof, count_latch=latch[s], csp_s=perf_counter()-start,
                                          count_effect=int(np.any(filtered[ix] != raw[ix], 1).sum()))
                stages[arm][s] = dict(**tape, raw=raw[ix], filtered=filtered[ix])
            works[arm]["count_s"] = count_seconds
    for s in sessions:
        a, b = (stages[arm][s] for arm in ARMS)
        assert works["stock"][str(s)]["covariance_sha"] == works["no_ft"][str(s)]["covariance_sha"]
        per_session[s] = decide(a, b)
    return rows, stages, per_session, works


def prefixes(ctx, stages, control, sessions):
    count, order, final = (ctx[k] for k in ("count", "order", "final"))
    checks = []
    for s in sessions:
        truncated = {}
        with limit_session(count, s):
            rows = count.C.order_rows(order, s, count.PROTO)
            assert len(rows) == 120
            for arm in ARMS:
                raw = ctx["matched"][arm]["raw"]
                filt, _ = count_filter(count, raw, order)
                got, proof = capture(final, ctx["windows"], raw, filt, rows)
                assert (proof["refits"], proof["fusions"]) == (5, 30)
                truncated[arm] = got
                np.testing.assert_array_equal(filt[rows], stages[arm][s]["filtered"][:120])
                for k in got:
                    np.testing.assert_array_equal(got[k], stages[arm][s][k][:120])
        for k, v in decide(truncated["stock"], truncated["no_ft"]).items():
            np.testing.assert_array_equal(v, control[s][k][:120])
        checks.append(str(s))
    return checks


def payload(rows, stages, choices):
    delivery = np.concatenate(list(rows.values()))
    canonical = np.argsort(delivery)
    out = dict(delivery=delivery, index=delivery[canonical])
    for arm in ARMS:
        for k in stages[arm][next(iter(rows))]:
            out[f"{arm}|{k}"] = np.concatenate([stages[arm][s][k] for s in rows])[canonical]
    for k in choices[next(iter(rows))]:
        out[k] = np.concatenate([choices[s][k] for s in rows])[canonical]
    return out


def main(order, smoke):
    start = perf_counter()
    release = sha(WORK/"source.sha256")
    if not smoke:
        s = json.loads((WORK/"smoke.json").read_text())
        assert s["passed"] and s["order"] == "full0" and s["source"] == release and not s["held_labels_opened"]
    load = {"full0": full0, "within0": within0}[order]
    with load() as ctx:
        count, whole = ctx["count"], tuple(ctx["count"].SESS)
        assert len(whole) == 14 and ctx["frozen_zero"] and ctx["warm"] == 9600
        with np.load(FEAT, allow_pickle=False) as z:
            people, visits = z["subject"], z["session"]

        def is_p01(s):
            ix = count.C.order_rows(order, s, count.PROTO)
            return set(people[ix]) == {"Dreyer2026Proteus/P01"} and set(visits[ix]) == {"1"}

        p01 = next(s for s in whole if is_p01(s))
        latched = count.latch(order)
        other = next(s for s in whole if s != p01 and latched[s] is not None and latched[s] < 120)
        proof_sessions = (p01, other, p01)
        selection = (p01, other) if smoke else whole
        rows, stages, choices, work = observe(ctx, selection)
        checks = prefixes(ctx, stages, choices, proof_sessions)
        if smoke:
            assert len(rows[p01]) == 190 and choices[p01]["eligible"].any()
            for arm in ARMS:
                p = work[arm][str(p01)]
                assert (p["refits"], p["fusions"]) == (9, 100)
            rows, choices = {p01:rows[p01]}, {p01:choices[p01]}
        arrays = payload(rows, stages, choices)
        if not smoke:
            np.testing.assert_array_equal(arrays["delivery"], ctx["delivery"])
            np.testing.assert_array_equal(arrays["index"], np.arange(2655))
            for arm in ARMS:
                w = [work[arm][str(s)] for s in whole]
                assert (sum(r["refits"] for r in w), sum(r["fusions"] for r in w)) == (126, 1395)
        assert not np.array_equal(arrays["stock|raw"], arrays["no_ft|raw"])
        assert any(info["count_effect"] for arm in work.values() for key,info in arm.items() if key != "count_s")
        n = "smoke" if smoke else order
        flags = dict(passed=True, order=order, source=release, held_labels_opened=False,
                     parent=ctx["proof"], archive_zero=ctx["zero"], raw_frozen_zero=True, stock_bitwise=True,
                     strict_past=True, no_recognition=True, warm_steps=9600, prefix=checks,
                     rows=len(arrays["index"]), sessions=len(rows), work=work,
                     activity={k:int(arrays[k].sum()) for k in ("eligible", "live", "sham_mask")})
        if not smoke:
            flags["output"] = write_npz(WORK/f"{order}.npz", arrays)
        output = WORK/f"{n}.json"
        write_receipt(output, **flags, seconds=perf_counter()-start)
    print("PASS F-C19", n, sha(output), flags["activity"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("order", choices=("full0", "within0"))
    parser.add_argument("mode", choices=("smoke", "complete"))
    args = parser.parse_args()
    assert args.mode != "smoke" or args.order == "full0"
    main(args.order, args.mode == "smoke")
