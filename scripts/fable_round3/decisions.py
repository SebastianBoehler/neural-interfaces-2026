"""Score Round 3 instruments and freeze both prespecified fit choices."""
import argparse
import json
from pathlib import Path

import numpy as np

from metrics import Session, arm, column, paired, summarize
from settings import DATA_SHA, D, FIT, FOLDS, HELD, LEAD_SHA, WEIGHTS, digest, streams
from blend import round3_log_pool as log_pool, round3_log_post as log_post

TAPE_KEYS = ("lead", "hard", "frozen", "pre_em", "hard_log", "team", "team_rb")


def load(run, phase, data, orders, names=()):
    cases = []
    identities = {}
    trace_only = phase == "trace"
    for fold in FOLDS:
        mode, person = fold.split("_")
        for first in range(0, len(orders), 2):
            chunk = orders[first:first + 2]
            filename = f"{fold}_{chunk[0]}.npz"
            with np.load(run / "trace" / filename, allow_pickle=False) as trace:
                assert str(trace["lead_sha"]) == LEAD_SHA and str(trace["data_sha"]) == DATA_SHA
                identity = {k: str(trace[k]) for k in ("weights_sha", "base_weights_sha", "source_sha")}
                assert all(len(sha) == 64 for sha in identity.values())
                assert identities.setdefault(fold, identity) == identity and tuple(trace["orders"]) == chunk
                extra = None
                if not trace_only:
                    extra = np.load(run / phase / filename, allow_pickle=False)
                    pinned = run / ("screen.json" if phase == "grid" else "select.json")
                    assert str(extra["decision_sha"]) == digest(pinned)
                    for key in (*identity, "lead_sha", "data_sha", "orders"):
                        np.testing.assert_array_equal(trace[key], extra[key])
                try:
                    for order in chunk:
                        for stream, rows in D.plans(data, order, streams(data, fold)).items():
                            visit = stream.split("|")[1]
                            key = f"{order}_{visit}"
                            np.testing.assert_array_equal(trace[f"rows_{key}"], rows)
                            values = {a: trace[f"{a}_{key}"] for a in TAPE_KEYS}
                            if extra is not None:
                                np.testing.assert_array_equal(extra[f"rows_{key}"], rows)
                                values.update({a: extra[f"{a}_{key}"] for a in names})
                            for a, v in values.items():
                                if a in {"pre_em", "hard_log", "team", "team_rb"}:
                                    assert v.shape == (len(rows), 3) and np.isfinite(v).all(), a
                                else:
                                    assert v.shape == rows.shape and np.isin(v, (0, 1, 2)).all(), a
                            np.testing.assert_array_equal(values["hard_log"].argmax(1), values["hard"])
                            cases.append(Session(mode, person, visit, order, data["y"][rows],
                                                 data["interface"][rows], values))
                finally:
                    if extra is not None:
                        extra.close()
    assert len(cases) == len(orders) * 18
    return cases, identities


def step0(cases, orders):
    causal, frozen = column("lead"), column("frozen")
    ranges = {"1–10": (1, 10), "11–20": (11, 20), "21–40": (21, 40), "41+": (41, 9999),
              "1–20": (1, 20), "1–30": (1, 30)}
    by_position = {}
    for name, (lo, hi) in ranges.items():
        def mask(c):
            return (c.age >= lo) & (c.age <= hi)
        by_position[name] = dict(causal=summarize(mask, causal, cases, orders),
                                 frozen=summarize(mask, frozen, cases, orders),
                                 delta=summarize(mask, frozen, cases, orders, mask, causal))
    gap = paired(cases, frozen, causal, orders)
    late = summarize(lambda c: c.age >= 41, causal, cases, orders, lambda c: c.age <= 20, causal)
    return dict(gap=gap, by_position=by_position, late_minus_first20=late,
                bet3_released=late["returning"]["mean"] >= 5 and gap["returning"]["mean"] >= 0.5)


def bet2_predict(case, pair):
    weights = np.where(case.changed, pair[1], pair[0])[:, None]
    return log_pool(case.tape["hard_log"], log_post(case.tape["team"]), weights).argmax(1)


def bet2_fit(cases):
    # Conditional summands are independent, so one sweep per weight gives the whole two-weight grid.
    grid = np.linspace(0, 1, 21)
    counts = {mode: sum(len(c.masks) for c in cases if c.mode == mode) for mode in ("new", "ret")}
    table = np.zeros((2, len(grid)))
    for case in cases:
        for wi, w in enumerate(grid):
            label = bet2_predict(case, (w, w))
            contribution = 50 * case.weights * case.correct(label) / counts[case.mode]
            table[:, wi] += [contribution[~case.changed].sum(), contribution[case.changed].sum()]
    a, b = min(((a, b) for a in range(21) for b in range(a, 21)),
               key=lambda p: (-round(table[0, p[0]] + table[1, p[1]], 10), abs(p[0]-4) + abs(p[1]-9), p))
    return [round(float(grid[i]), 2) for i in (a, b)]


def screen(cases):
    early = lambda c: c.age <= 15
    rb, raw = arm("team_rb"), arm("team")
    early_delta = summarize(early, rb, cases, FIT, early, raw)
    same, changed = lambda c: ~c.changed, lambda c: c.changed
    hard, pre = column("hard"), arm("pre_em")
    table = {name: {key: summarize(mask, predict, cases, FIT) for key, predict in
                    (("base_final", hard), ("pre_em", pre), ("team_b", raw))}
             for name, mask in (("unchanged", same), ("changed", changed))}
    gap = summarize(same, hard, cases, FIT, changed, hard)
    hedge = summarize(changed, raw, cases, FIT, changed, hard)
    bet2_go = gap["j"] >= 10 and hedge["j"] > 0
    choice = bet2_fit(cases) if bet2_go else None
    return dict(step0_fit=step0(cases, FIT), bet1_start=15 if early_delta["j"] < 0 else 1,
                team_b_early_delta=early_delta, bet2_table=table, bet2_gap=gap, bet2_hedge=hedge,
                bet2_weights=choice)


def select(cases, report):
    names = [f"b1_{w:.3f}" for w in WEIGHTS] + ["geometric_25"]
    contrasts = {name: paired(cases, column(name), column("lead"), FIT) for name in names}
    weight = min(WEIGHTS, key=lambda w: (-round(contrasts[f"b1_{w:.3f}"]["j"], 10),
                                         round(abs(w - 0.25), 4), w))
    return dict(bet1_start=report["bet1_start"], bet1_weight=weight, bet1_fit=contrasts,
                bet2_weights=report["bet2_weights"])


def confirm(cases, frozen):
    lead = column("lead")
    bet1 = paired(cases, column(f"b1_{frozen['bet1_weight']:.3f}"), lead, HELD)
    bet1["go"] = all(bet1[k]["mean"] >= 0.2 for k in ("new", "returning"))
    raw, rb = arm("team"), arm("team_rb")

    def delayed(c):
        return np.where(c.age < frozen["bet1_start"], raw(c), rb(c))

    bet2 = None
    if frozen["bet2_weights"] is not None:
        bet2 = paired(cases, lambda c: bet2_predict(c, frozen["bet2_weights"]), lead, HELD)
        points = [bet2[k]["mean"] for k in ("new", "returning")]
        bet2["go"] = max(points) >= 0.2 and min(points) >= 0
    return dict(step0_held=step0(cases, HELD), team_b_raw_rb=paired(cases, rb, raw, HELD),
                team_b_chosen_start=paired(cases, delayed, raw, HELD), bet1=bet1, bet2=bet2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("screen", "select", "confirm"))
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--run", required=True, type=Path)
    a = p.parse_args()
    data = D.load(a.data)
    if digest(a.data) != DATA_SHA:
        raise ValueError("unexpected PROTEUS export SHA-256")
    if a.stage == "screen":
        cases, identity = load(a.run, "trace", data, FIT)
        result = screen(cases)
    elif a.stage == "select":
        prev = json.loads((a.run / "screen.json").read_text())
        names = [f"b1_{w:.3f}" for w in WEIGHTS] + ["geometric_25"]
        cases, identity = load(a.run, "grid", data, FIT, names)
        assert prev["folds"] == identity
        result = select(cases, prev)
    else:
        prev = json.loads((a.run / "select.json").read_text())
        cases, identity = load(a.run, "held", data, HELD, [f"b1_{prev['bet1_weight']:.3f}"])
        assert prev["folds"] == identity
        result = confirm(cases, prev)
    result.update(folds=identity, data_sha=DATA_SHA)
    with (a.run / f"{a.stage}.json").open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print(a.stage, digest(a.run / f"{a.stage}.json"), flush=True)


if __name__ == "__main__":
    main()
