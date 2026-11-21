"""Freeze the neighbour vote on eight orders and compare both arms on eight held orders."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from new_cohort_vote import GRID, START_WINDOW, count_terminal, vote
from returning_diet_report import digest, load_bank, person_delta, predict
from returning_em_report import ORDERS
from oracles import freeze
from tapes import DATA_SHA, PEOPLE, joint, metadata, paired, pinned, read_bank, score

PARENT = "42fb63cf1ae9f54cf48315b00894b3788bc65400e75e06bd208ffe3753c43b07"
ROSTER = "ba_six"
FIELDS = ("hard", "team", "own", "count", "csp", "csp_active", "manas", "terminal", "label", "recognised")
OLD_STARTS = (71800, 71802, 71808, 71810)
PROTOCOL = dict(parent=PARENT, data=DATA_SHA, grid=[list(x) for x in GRID], start=START_WINDOW,
                held=list(ORDERS["held"]), bootstrap="20,000 paired people-nested draws")


def archive(args, data, phase, orders):
    diet_cases, diet_hashes = load_bank(args.archive, args.diet, data, phase)
    diet = {case.key: case for case in diet_cases}
    paths = set()

    def read(fold, order, key):
        first = order - order % 2
        folder = args.archive / "capture" if first in OLD_STARTS else args.extension
        path = folder / f"{ROSTER}-{fold}-{first}.npz"
        paths.add(path)
        rows, expected, receipt = pinned(args.archive / "leo", fold, ROSTER, order, key)
        ledger = json.loads(receipt)
        last_push = args.archive / "leo" / phase / ROSTER / fold / f"{first}.npz"
        with np.load(last_push, allow_pickle=False) as zero:
            if str(zero["receipt"]) != receipt:
                raise ValueError(f"wrong Last Push reference in {last_push}")
            np.testing.assert_array_equal(zero[f"rows_{key}"], rows)
            np.testing.assert_array_equal(zero[f"w25_{key}"], expected)
        with np.load(path, allow_pickle=False) as tape:
            if str(tape["receipt"]) != receipt or ledger["fold"] != fold or ledger["data_sha"] != DATA_SHA:
                raise ValueError(f"wrong ba_six capture receipt in {path}")
            np.testing.assert_array_equal(rows, tape[f"rows_{key}"])
            np.testing.assert_array_equal(expected, tape[f"label_{key}"])
            fields = {name: tape[f"{name}_{key}"] for name in FIELDS}
        cohort, person = fold.split("_")
        control = diet[cohort, order, person, key.split("_")[1]]
        np.testing.assert_array_equal(control.rows, rows)
        np.testing.assert_array_equal(control.scores["native"], expected)
        age = int(control.scores["rec"][0])
        recognised = (np.arange(len(rows)) + 1 >= age) if age != -1 else np.zeros(len(rows), dtype=bool)
        np.testing.assert_array_equal(fields["recognised"], recognised)
        fields["quarter"] = control.scores["quarter"]
        np.testing.assert_array_equal(fields["quarter"][~recognised], expected[~recognised])
        return rows, fields

    bank = read_bank(data, orders, read)
    return bank, {p.name: digest(p) for p in sorted(paths)}, diet_hashes


def vote_key(k, weight):
    return f"A:k{k},w{weight}"


def decide(bank, settings):
    coverage = Counter()
    for case in bank:
        s = case.scores
        active = ~s["recognised"]
        native = s["label"]
        for key in ("hard", "team", "terminal", "count", "own", "csp", "manas"):
            if s[key].shape != (len(native), 3) or (s[key] < 0).any():
                raise ValueError(f"invalid {key} array for {case.key}")
        np.testing.assert_array_equal(native, s["terminal"].argmax(1))
        csp_zero = count_terminal(s, np.zeros_like(active))
        np.testing.assert_allclose(csp_zero, s["terminal"], rtol=0, atol=1e-7)
        np.testing.assert_array_equal(csp_zero.argmax(1), native)
        features = np.column_stack((s["hard"], s["team"])).astype(np.float64)
        terminal = s["terminal"].astype(np.float64)
        zero = vote(features, terminal, s["recognised"], 7, 0)
        np.testing.assert_array_equal(zero.argmax(1), native)
        simulations = {"B": count_terminal(s, active)}
        position = np.arange(len(native)) + 1
        for k, weight in settings:
            name = vote_key(k, weight)
            full = vote(features, terminal, s["recognised"], k, weight)
            prefix = min(len(native), 67)
            short = vote(features[:prefix], terminal[:prefix], s["recognised"][:prefix], k, weight)
            np.testing.assert_array_equal(full[:prefix], short)
            np.testing.assert_array_equal(full[~active | (position < START_WINDOW)],
                                          zero[~active | (position < START_WINDOW)])
            simulations[name] = full
            coverage[f"{name}_active"] += int(np.count_nonzero(full.argmax(1) != native))
        cut = min(len(native), 67)
        prefix_scores = {k: v[:cut] for k, v in s.items()}
        np.testing.assert_array_equal(simulations["B"][:cut], count_terminal(prefix_scores, active[:cut]))
        coverage["B_active"] += int(np.count_nonzero(np.any(simulations["B"] != csp_zero, axis=1)))
        coverage[f"{case.cohort}_predictions"] += len(native)
        coverage[f"{case.cohort}_post_latch"] += int((~active).sum())
        coverage[f"{case.cohort}_recognised_streams"] += int((~active).any())
        for name, probabilities in simulations.items():
            labels = probabilities.argmax(1)
            np.testing.assert_array_equal(labels[~active], native[~active])
            s[name] = np.where(active, labels, s["quarter"])
    if not all(coverage[f"{k}_active"] > 0 for k in ("B", *(vote_key(*x) for x in settings))):
        raise ValueError("at least one active candidate was a no-op")
    return coverage


def scores(bank, arm, orders):
    reference = predict("quarter")
    candidate = predict(arm)
    intervals = paired(bank, candidate, reference, orders, nested_orders=True)
    point, lower, _ = intervals["J"]
    return dict(delta=intervals, people={p: person_delta(bank, orders, arm, p, reference="quarter")
                                         for p in PEOPLE["ret"]},
                changed={c: sum(int(np.count_nonzero(candidate(d) != reference(d)))
                                for d in bank if d.cohort == c) for c in PEOPLE},
                pre_latch_changes_in_recognised_streams=sum(
                    int(np.count_nonzero(candidate(d) != reference(d))) for d in bank if d.scores["recognised"].any()),
                passes=bool(point >= 0.15 and lower > 0 and intervals["ret"][0] >= -0.10))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("smoke", *ORDERS))
    for name in ("data", "archive", "extension", "diet", "out"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    data = metadata(a.data)
    if a.phase == "smoke":
        # Verify the scorer on its real input format, without computing any accuracy.
        bank, _, _ = archive(a, data, "fit", (71800, 71801))
        print("PASS archive/zero/prefix/active checks", dict(decide(bank, GRID)))
        return
    orders = ORDERS[a.phase]
    fit_path = a.out / "fit.json"
    if a.phase == "held":
        fit = json.loads(fit_path.read_text())
        if fit["orders"] != list(ORDERS["fit"]) or fit["protocol"] != PROTOCOL:
            raise ValueError("selection used another protocol or split")
        selected = tuple(fit["selected"])
        if selected not in GRID:
            raise ValueError("selection is outside the registered grid")
    variants = (selected,) if a.phase == "held" else GRID
    bank, hashes, diet_hashes = archive(a, data, a.phase, orders)
    coverage = decide(bank, variants)
    names = tuple(vote_key(*x) for x in variants)
    arms = {key: scores(bank, key, orders) for key in (*names, "B")}
    if a.phase == "fit":
        best = max(arms[name]["delta"]["J"][0] for name in names)
        selected = next(x for x in GRID if arms[vote_key(*x)]["delta"]["J"][0] >= best - 1e-10)
    result = dict(orders=orders, protocol=PROTOCOL, selected=selected, coverage=coverage,
                  baseline={**{c: score(bank, predict("quarter"), c) for c in PEOPLE},
                            "J": joint(bank, predict("quarter"))}, arms=arms,
                  capture_sha256=hashes, diet_sha256=diet_hashes)
    if a.phase == "held":
        result["fit_sha256"] = digest(fit_path)
        result["promote"] = max((key for key, value in arms.items() if value["passes"]),
                                key=lambda key: arms[key]["delta"]["J"][0], default=None)
    a.out.mkdir(parents=True, exist_ok=True)
    freeze(a.out / f"{a.phase}.json", result)
    print(json.dumps({k: v for k, v in result.items() if not k.endswith("_sha256")}, indent=2))


if __name__ == "__main__":
    main()
