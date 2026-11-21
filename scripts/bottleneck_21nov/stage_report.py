"""Attribute adjacent native decision changes and check posterior calibration."""
from collections import defaultdict

import numpy as np

from errors import table
from tapes import PEOPLE, POSITIONS, score, subsets

STAGES = ("raw", "hard", "blend", "routed", "rebalance", "session_em", "tsem", "own",
          "count", "csp", "terminal")
NAMES = ("Raw mixer", "Hard mix", "Team B blend", "Personal route", "Rebalance", "SessionEM",
         "TSEM", "Own-space", "Count filter", "CSP", "MANAS")


def pred(stage):
    return lambda case: case.scores[stage].argmax(1)


def flips(cases, old, new, cohort, **filters):
    gain = loss = 0
    for case, mask in subsets(cases, cohort=cohort, **filters):
        prior, after = (get(case)[mask] == case.y[mask] for get in (old, new))
        gain += np.count_nonzero(~prior & after)
        loss += np.count_nonzero(prior & ~after)
    return f"{gain}/{loss}"


def stage_report(cases, interfaces, has_manas):
    rows, stratified = [], []
    dims = [{}, *({"interface": i} for i in interfaces), *({"position": i} for i in range(len(POSITIONS)))]
    for index, stage in enumerate(STAGES[: None if has_manas else -1]):
        before = STAGES[max(0, index - 1)]
        old, new = pred(before), pred(stage)
        outcome = []
        for cohort in PEOPLE:
            before_ba, after_ba = (score(cases, get, cohort) for get in (old, new))
            outcome.extend((f"{before_ba:.2f} → {after_ba:.2f}" if index else f"{after_ba:.2f}",
                            flips(cases, old, new, cohort) if index else "—"))
            after = [score(cases, new, cohort, **d) for d in dims]
            if index:
                delta = [v - score(cases, old, cohort, **d) for v, d in zip(after, dims, strict=True)]
                entries = [f"{v:+.2f} ({flips(cases, old, new, cohort, **d)})" if "position" in d else f"{v:+.2f}"
                           for v, d in zip(delta, dims, strict=True)]
            else:
                entries = [f"{v:.2f}%" for v in after]
            stratified.append((cohort, NAMES[index], *entries))
        rows.append((NAMES[index], *outcome))
    return "\n\n".join([
        table(["Stage", "New BA%", "New flips +/−", "Returning BA%", "Ret. flips +/−"], rows),
        table(["Cohort", "Stage", "All Δ", *interfaces, *POSITIONS], stratified),
    ])


def calibration(cases, interfaces):
    rows = []
    for cohort in PEOPLE:
        for interface in (*interfaces, None):
            y, p, w = [], [], []
            for case, mask in subsets(cases, cohort=cohort, interface=interface):
                y.extend(case.y[mask])
                p.extend(case.scores["terminal"][mask])
                w.extend(case.weights[mask])
            y, p, w = np.array(y), np.array(p), np.array(w)
            w /= w.sum()
            onehot = np.eye(3)[y]
            predicted = p.argmax(1)
            hit = y == predicted
            confidence = p.max(1)
            group = np.minimum((confidence * 10).astype(int), 9)
            ece = np.abs(np.bincount(group, weights=w * (confidence - hit), minlength=10)).sum()
            class_error = w @ (p - onehot)
            label_error = w @ (np.eye(3)[predicted] - onehot)
            rows.append((cohort, interface or "Both", f"{100 * (w @ hit):.2f}",
                         f"{100 * (w @ confidence):.2f}", f"{100 * ece:.2f}",
                         f"{w @ -np.log(np.maximum(p[np.arange(len(p)), y], 1e-12)):.3f}",
                         f"{w @ ((p - onehot) ** 2).sum(1):.3f}",
                         "/".join(f"{x:+.1f}" for x in class_error * 100),
                         "/".join(f"{x:+.1f}" for x in label_error * 100)))
    return table(["Cohort", "Interface", "BA%", "Confidence%", "ECE pp", "NLL", "Brier",
                  "Probability bias 0/1/2 pp", "Decision bias 0/1/2 pp"], rows)


def recognition(cases):
    rows = []
    observed = defaultdict(list)
    for case in cases:
        gate = case.scores["gate_person"]
        active = case.scores["recognised"]
        if active.any():
            first = int(np.argmax(active))
            if not gate[first]:
                raise ValueError("recognition latch without an active gate")
            outcome = ("correct" if gate[first] == int(case.person[1:]) and case.cohort == "ret"
                       else "false/wrong")
            observed[case.cohort, case.person].append((first + 1, outcome))
        else:
            observed[case.cohort, case.person].append((0, "none"))
    for (cohort, person), events in observed.items():
        ages = [age for age, _ in events if age > 0]
        if cohort == "new" and not ages:
            continue
        rows.append((cohort, person, len(events), sum(flag == "correct" for _, flag in events),
                     sum(flag == "false/wrong" for _, flag in events),
                     f"{min(ages)} / {np.median(ages):g} / {max(ages)}" if ages else "—"))
    return table(["Cohort", "Person", "Streams", "Correct", "False/wrong", "First min/median/max"], rows)
