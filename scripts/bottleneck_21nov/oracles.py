"""Retrospective ceilings on frozen score streams and the terminal bias grid."""
from collections import defaultdict
from itertools import product
from pathlib import Path
import json

import numpy as np

from errors import fmt_ci, table
from stage_report import pred
from tapes import PEOPLE, paired, score

BASELINE = "terminal"
GRID = ((0.0, 0.0), *sorted(
    ((x, y) for x, y in product((-0.20, -0.10, 0.0, 0.10, 0.20), repeat=2) if (x, y) != (0, 0)),
    key=lambda pair: (sum(x * x for x in pair), pair)))
BLEND = np.arange(0, 21) / 20


def members(case):
    s = case.scores
    groups = [s[k][:, None] for k in ("core1", "core2", "rest")]
    groups += [s[k] for k in ("net_members", "hard_actors", "team_members")]
    if "manas" in s:
        groups.append(s["manas"][:, None])
    return np.concatenate(groups, axis=1)


def member_names(case):
    scores = case.scores
    names = ["core1", "core2", "rest"]
    names.extend(f"raw net {i}" for i in scores["net_ids"][0])
    names.extend(f"hard actor {i + 1}" for i in range(scores["hard_actors"].shape[1]))
    names.extend(f"Team B member {i + 1}" for i in range(scores["team_members"].shape[1]))
    if "manas" in scores:
        names.append("MANAS")
    return names


def member_report(cases):
    names = member_names(cases[0])
    weights, hits = defaultdict(float), defaultdict(lambda: np.zeros(len(names)))
    for case in cases:
        if member_names(case) != names:
            raise ValueError("inconsistent member IDs across captured folds")
        weights[case.cohort] += case.weights.sum()
        hits[case.cohort] += case.weights @ (members(case).argmax(2) == case.y[:, None])
    rows = []
    for c in PEOPLE:
        score = 100 * hits[c] / weights[c]
        best = np.argsort(-score, kind="stable")[:3]
        rows.append((c, *(f"{names[i]} {score[i]:.2f}%" for i in best)))
    return table(["Cohort", "Best member", "Second", "Third"], rows)


def endpoints(case):
    s = case.scores
    return np.concatenate([np.stack([s[k] for k in ("raw", "hard", "team", "blend", "routed",
                                                      "rebalance", "session_em", "tsem", "own", "count",
                                                      "csp")], axis=1), members(case)], axis=1)


def best_member(cases):
    bank = defaultdict(list)
    for case in cases:
        hit = members(case).argmax(2) == case.y[:, None]
        terminal = pred(BASELINE)(case) == case.y
        for iface in np.unique(case.interface):
            mask = case.interface == iface
            w = case.weights[mask]
            results = w @ hit[mask]
            reference = w @ terminal[mask]
            any_member_or_terminal_correct = w @ (hit.any(1) | terminal)[mask]
            bank[case.cohort, case.person, case.visit, iface].append(
                np.r_[results, reference, any_member_or_terminal_correct])
    result = defaultdict(list)
    for cell, values in bank.items():
        avg = np.mean(values, axis=0)
        best, ref, any_correct = avg[:-2].max(), avg[-2], avg[-1]
        result[cell[0]].append([best - ref, max(0, best - ref), any_correct - ref])
    return {cohort: 100 * np.mean(rows, axis=0) for cohort, rows in result.items()}


def pooling(cases, interfaces):
    totals, weights = {}, {}
    for case in cases:
        s = case.scores
        source = endpoints(case)
        ref = s["terminal"][:, None, None, :]
        w = BLEND[None, None, :, None]
        candidates = ((1 - w) * ref + w * source[:, :, None]).argmax(3)
        correct = candidates == case.y[:, None, None]
        np.testing.assert_array_equal(candidates[:, 0, 0], s["label"])
        for iface in interfaces:
            mask = case.interface == iface
            cell = (case.cohort, iface)
            m = case.weights[mask]
            weighted = np.einsum("n,nbw->bw", m, correct[mask])
            totals[cell] = totals.get(cell, 0) + weighted
            weights[cell] = weights.get(cell, 0) + m.sum()
    per_interface = {i: np.mean([100 * totals[c, i] / weights[c, i] for c in PEOPLE], axis=0) for i in interfaces}
    grid = np.mean(list(per_interface.values()), axis=0)
    best = np.unravel_index(np.argmax(grid + (BLEND == 0) * 1e-10), grid.shape)
    endpoint_names = ["raw", "hard", "Team B", "blend", "routed", "Rebalance", "SessionEM", "TSEM",
                      "own-space", "count", "CSP", *member_names(cases[0])]
    reference = grid[0, 0]
    rows = []

    def explain(idx):
        return f"{BLEND[idx[1]]:.0%} {endpoint_names[idx[0]]}" if idx[1] else "native terminal"

    rows.append(("Global fixed hindsight pool", f"{grid[best] - reference:+.2f}", explain(best)))
    settings = []
    for i in interfaces:
        a = per_interface[i]
        idx = np.unravel_index(np.argmax(a + (BLEND == 0) * 1e-10), a.shape)
        settings.append(f"{i} {explain(idx)}")
    oracle = np.mean([a.max() for a in per_interface.values()])
    rows.append(("Interface-known hindsight pool", f"{oracle - reference:+.2f}", " · ".join(settings)))
    rows.append(("Increment from interface knowledge", f"{oracle - grid[best]:+.2f}", "Same pool family"))
    return rows


def true_prior(case):
    p = case.scores[BASELINE].copy()
    for i in np.unique(case.interface):
        mask = case.interface == i
        counts = np.bincount(case.y[mask], minlength=3)
        if (counts == 0).any():
            raise ValueError("true prior contains an absent class")
        p[mask] /= counts / counts.sum()
    return p.argmax(1)


def offset(pair):
    log_bias = np.array((0.0, *pair))

    def candidate(case):
        p = case.scores[BASELINE]
        return (np.log(np.maximum(p, 1e-30)) + log_bias).argmax(1)
    return candidate


def prior_grid(cases):
    """Upper envelope for these offsets, selected in hindsight per official cell."""
    grid = np.array([(0.0, *v) for v in GRID])
    bank = defaultdict(list)
    for case in cases:
        p = np.log(np.maximum(case.scores[BASELINE], 1e-30))
        correct = (p[:, None, :] + grid[None]).argmax(2) == case.y[:, None]
        for i in np.unique(case.interface):
            mask = case.interface == i
            bank[case.cohort, case.person, case.visit, i].append(case.weights[mask] @ correct[mask])
    gains = defaultdict(list)
    for cell, values in bank.items():
        scores = np.mean(values, axis=0)
        gains[cell[0]].append(100 * (scores.max() - scores[0]))
    return {cohort: np.mean(v) for cohort, v in gains.items()}


def ceiling_report(cases, orders, interfaces):
    best = best_member(cases)
    rows = []
    for index, name in enumerate(("Best single member chosen per cell", "Best member or keep terminal, per cell",
                                  "Any correct member or terminal, per window")):
        new, ret = (best[c][index] for c in PEOPLE)
        rows.append((name, f"{new:+.2f}", f"{ret:+.2f}", f"{(new + ret) / 2:+.2f}"))
    controls = (("True cell class prevalence correction", true_prior),
                ("True identity from delivery 1 (full ramp)", lambda c: c.scores["oracle_recognition"]))
    for name, candidate in controls:
        result = paired(cases, candidate, pred(BASELINE), orders)
        rows.append((name, *(fmt_ci(result[c]) for c in ("new", "ret", "J"))))
    grid = prior_grid(cases)
    rows.append(("Best grid class offsets per cell, hindsight", *(f"{grid[c]:+.2f}" for c in PEOPLE),
                 f"{np.mean([grid[c] for c in PEOPLE]):+.2f}"))
    by_person = []
    for person in PEOPLE["ret"]:
        selected = [c for c in cases if c.person == person]
        d = score(selected, controls[-1][1], "ret") - score(selected, pred(BASELINE), "ret")
        by_person.append(f"{person} {d:+.2f} pp")
    return "\n\n".join([
        member_report(cases),
        table(["Oracle / diagnostic", "New Δ pp", "Ret. Δ pp", "50/50 Δ pp"], rows),
        table(["Terminal pool", "50/50 Δ pp", "Selected blend"], pooling(cases, interfaces)),
        "Returning forced-recognition deltas are " + " and ".join(by_person) + ".",
    ])


def verify_zero(cases):
    for case in cases:
        np.testing.assert_array_equal(case.scores["label"], offset((0.0, 0.0))(case))


def freeze(path, selection):
    with Path(path).open("x") as f:
        json.dump(selection, f, indent=2)
        f.write("\n")
