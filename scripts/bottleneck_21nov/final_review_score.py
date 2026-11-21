"""Select on four fit orders and score only the frozen doses on four held orders."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np

from final_review_emissions import BALANCE_DOSES, COUNT_DOSES, balance_output
from oracles import freeze
from returning_em_report import load_cases
from tapes import joint, metadata, paired, pinned, score, stages
from terminal import check

ORDERS = {"fit": tuple(range(71800, 71804)), "held": tuple(range(71808, 71812))}
SPECS = {"count": COUNT_DOSES, "balance": BALANCE_DOSES}
ROSTER = "ba_six"


def stream_balance(raw, flags, strength):
    history = []
    return np.stack([balance_output(p, history, seen, strength) for p, seen in zip(raw, flags, strict=True)])


def emissions(case, legacy, fresh, phase):
    base = legacy.scores["half"]
    flags = legacy.scores["rec"] > 0
    flags = flags & (np.arange(len(base)) >= legacy.scores["rec"] - 1)
    np.testing.assert_array_equal(case.rows, legacy.rows)
    np.testing.assert_array_equal(case.scores["recognised"], flags)
    out = {("count", strength): base for strength in COUNT_DOSES}
    if flags.any():
        fold = f"{case.cohort}_{case.person}"
        key = f"{case.order}_{case.visit}"
        _, _, receipt = pinned(fresh["work"] / "leo", fold, ROSTER, case.order, key)
        with np.load(fresh["captures"] / f"{fold}-{phase}.npz", allow_pickle=False) as tape:
            if str(tape["receipt"]) != receipt:
                raise ValueError(f"capture and pinned fixture differ: {fold}/{key}")
            np.testing.assert_array_equal(tape[f"rows_{key}"], case.rows)
            np.testing.assert_array_equal(tape[f"label_{key}"], base)
            np.testing.assert_array_equal(tape[f"recognised_{key}"], flags)
            raw = tape[f"terminal_{key}"]
            for strength in COUNT_DOSES[1:]:
                out["count", strength] = tape[f"count{strength:g}_{key}"]
    else:
        np.testing.assert_array_equal(case.scores["label"], base)
        raw = case.scores["terminal"]
    if raw.shape != (len(base), 3) or not np.isfinite(raw).all() or np.min(raw) < 0:
        raise ValueError("invalid em_half terminal posterior")
    np.testing.assert_array_equal(raw.argmax(1), base)
    for strength in COUNT_DOSES:
        np.testing.assert_array_equal(out["count", strength][~flags], base[~flags])
    masks = np.arange(len(base)) < 19
    for strength in BALANCE_DOSES:
        q = stream_balance(raw, flags, strength)
        np.testing.assert_array_equal(q[masks | flags], raw[masks | flags])
        if strength == 0:
            np.testing.assert_array_equal(q, raw)
        out["balance", strength] = q.argmax(1)
    q = stream_balance(raw[:67], flags[:67], 1.0)
    np.testing.assert_array_equal(q.argmax(1), out["balance", 1.0][:67])
    return base, out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=ORDERS)
    for name in ("data", "work", "captures", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--smoke", action="store_true", help="verify inputs, identity and prefix only")
    a = parser.parse_args()
    phase, orders = a.phase, ORDERS[a.phase]
    info = metadata(a.data)
    cases = stages(info, a.work, ROSTER, orders)
    check(cases, ROSTER)
    legacy = {c.key: c for c in load_cases(a.work, info, ROSTER, phase)}
    cfg = {"work": a.work, "captures": a.captures}
    predictions = {c.key: emissions(c, legacy[c.key], cfg, phase) for c in cases}
    reference = lambda c: predictions[c.key][0]
    if a.smoke:
        for name, strengths in SPECS.items():
            if not any(np.any(v[1][name, strengths[-1]] != v[0]) for v in predictions.values()):
                raise ValueError(f"{name} did not reach any fit decision")
        print(f"PASS {len(cases)} sessions: canonical labels, zero, activation, prefix and session reset")
        return
    a.out.mkdir(parents=True, exist_ok=True)
    baseline = joint(cases, reference)
    fit_file = a.out / "fit.json"
    if phase == "fit":
        arms = {}
        for name, strengths in SPECS.items():
            values = [
                (s, joint(cases, lambda c: predictions[c.key][1][name, s]) - baseline)
                for s in strengths
            ]
            best = max(point for _, point in values)
            selected = next(s for s, point in values if point >= best - 1e-10)
            arms[name] = {"selected": selected, "fit_gain_pp": dict(values)}
        result = {"orders": orders, "baseline_50_50": baseline, "arms": arms}
        freeze(fit_file, result)
    else:
        with fit_file.open("rb") as source:
            fit_sha = file_digest(source, "sha256").hexdigest()
        fit = json.loads(fit_file.read_text())
        if tuple(fit["orders"]) != ORDERS["fit"]:
            raise ValueError("wrong fit freeze")
        arms = {}
        for name, strengths in SPECS.items():
            strength = fit["arms"][name]["selected"]
            if strength not in strengths:
                raise ValueError("frozen dose outside the registered grid")
            candidate = lambda c: predictions[c.key][1][name, strength]
            delta = paired(cases, candidate, reference, orders)
            changed = {cohort: sum(np.count_nonzero(candidate(c) != reference(c))
                                   for c in cases if c.cohort == cohort) for cohort in ("new", "ret")}
            returning = {}
            for person in ("P04", "P10"):
                subset = [c for c in cases if c.cohort == "ret" and c.person == person]
                returning[person] = score(subset, candidate, "ret") - score(subset, reference, "ret")
            point, low, _ = delta["J"]
            arms[name] = {"strength": strength, "delta_pp_80": delta, "changed": changed,
                          "returning_people_pp": returning,
                          "pass": bool(point >= 0.15 and low > 0 and delta["new"][0] >= -0.10)}
        result = {"orders": orders, "baseline_50_50": baseline, "fit_sha256": fit_sha, "arms": arms}
        freeze(a.out / "held.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
