"""One-factor decisions after every stateful stage has consumed the current window."""
import numpy as np

from errors import fmt_ci
from oracles import GRID, offset, verify_zero
from stage_report import pred
from tapes import PEOPLE, joint, paired

MANAS = (0.10, 0.05, 0.15, 0.0, 0.20)
CSP = (1.0, 0.5, 1.5)


def mix_manas(case, score, weight=0.10):
    return (1 - weight) * score + weight * case.scores["manas"] if "manas" in case.scores else score


def manas(weight):
    return lambda case: mix_manas(case, case.scores["csp"], weight)


def csp(multiplier):
    def score(case):
        base = case.scores["count"]
        active = case.scores["csp_active"]
        q = base.copy()
        log_base = np.log(np.clip(base[active], 1e-6, 1))
        relative_evidence = np.log(case.scores["csp"][active]) - log_base
        logits = log_base + multiplier * relative_evidence
        weights = np.exp(logits - logits.max(axis=1, keepdims=True))
        q[active] = weights / weights.sum(axis=1, keepdims=True)
        return mix_manas(case, q)
    return score


def candidate(score):
    return lambda case: score(case).argmax(1)


def arms(roster):
    tests = {"Class 1/2 log odds": (GRID, offset), "CSP evidence multiplier": (CSP, lambda w: candidate(csp(w)))}
    if roster == "ba_six":
        tests["MANAS weight"] = (MANAS, lambda w: candidate(manas(w)))
    return tests


def check(cases, roster):
    verify_zero(cases)
    for case in cases:
        controls = (csp(1.0), manas(0.10)) if roster == "ba_six" else (csp(1.0),)
        for control in controls:
            q = control(case)
            np.testing.assert_allclose(q, case.scores["terminal"], rtol=0, atol=1e-7)
            np.testing.assert_array_equal(q.argmax(1), case.scores["label"])


def fit(cases, roster):
    check(cases, roster)
    result = {}
    for name, (grid, build) in arms(roster).items():
        points = [(setting, joint(cases, build(setting))) for setting in grid]
        best = max(value for _, value in points)
        selection, point = next((setting, value) for setting, value in points if value >= best - 1e-10)
        result[name] = {"selection": selection, "fit_gain_pp": point - points[0][1]}
    return result


def held(cases, roster, selections, orders):
    check(cases, roster)
    rows = []
    for name, (grid, build) in arms(roster).items():
        selection = selections[name]["selection"]
        if isinstance(selection, list):
            selection = tuple(selection)
        if selection not in grid:
            raise ValueError(f"frozen setting outside registered grid: {name} {selection}")
        fn = build(selection)
        result = paired(cases, fn, pred("terminal"), orders)
        changed = {c: sum(np.count_nonzero(fn(case) != case.scores["label"]) for case in cases if case.cohort == c)
                   for c in PEOPLE}
        label = "/".join(f"{x:+.2f}" for x in selection) if isinstance(selection, tuple) else f"{selection:g}"
        rows.append((roster, name, label, f"{selections[name]['fit_gain_pp']:+.2f}",
                     *(fmt_ci(result[c]) for c in ("new", "ret", "J")), f"{changed['new']}/{changed['ret']}"))
    return rows
