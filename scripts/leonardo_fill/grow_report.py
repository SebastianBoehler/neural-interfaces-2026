"""Freeze one timing-eligible growth arm and assess its held-out full-Solver gain."""
import argparse
import json
from pathlib import Path

import numpy as np

from grow_fixture import ARMS, DATA_SHA, HELD, PARENT_SHA, REFERENCE, SCREEN
from grow_replay import TIMING_FOLDS
from report import V, contrasts, load, print_contrasts

LIMIT_MS = 8.0
MIN_GAIN = 0.20


def time_results(root):
    out = {}
    for arm in ARMS[1:]:
        places, count, seconds = {}, 0, 0.0
        for fold in TIMING_FOLDS:
            obj = json.loads((root / "timing" / f"{fold}-{arm}.json").read_text())
            if obj["arm"] != arm or obj["fold"] != fold or "A100" not in obj["device"] or len(obj["pairs"]) != 2:
                raise ValueError(f"invalid paired A100 timing for {fold} {arm}")
            pair_ms = []
            for pair in obj["pairs"]:
                delta = sum(pair[arm][k] - pair[REFERENCE][k] for k in ("warm_s", "setup_s"))
                if pair["windows"] <= 0 or not np.isfinite(delta):
                    raise ValueError(f"invalid timing pair for {fold} {arm}: {pair}")
                count += pair["windows"]
                seconds += delta
                pair_ms.append(1000 * delta / pair["windows"])
            places[fold] = dict(device=obj["device"], pair_ms=pair_ms,
                                added_load_s=obj["load_s"][arm] - obj["load_s"][REFERENCE])
        out[arm] = dict(ms_per_window=1000 * seconds / count, windows=count, placements=places)
    return out


def verified_cells(root, phase, arms, orders, data):
    cells, receipts = load(root / phase, arms, orders, data)
    for fold, rows in receipts.items():
        expected = json.loads((root / "fixtures" / fold / "receipt.json").read_text())
        if any(actual != expected[arm] for arm, actual in rows.items()):
            raise ValueError(f"{fold}: changed staged fixture or source receipt")
        if any(r["lead_sha"] != PARENT_SHA or r["data_sha"] != DATA_SHA for r in rows.values()):
            raise ValueError(f"{fold}: wrong lead or data")
    return cells


def both_positive(result):
    return all(result[k]["mean"] > 0 for k in ("new", "returning"))


def select(root, data):
    timing = time_results(root)
    cells = verified_cells(root, "screen", ARMS, SCREEN, data)
    comparisons = contrasts(cells, [(arm, REFERENCE) for arm in ARMS[1:]], SCREEN)
    eligible = [arm for arm, t in timing.items()
                if t["ms_per_window"] <= LIMIT_MS and both_positive(comparisons[f"{arm} - {REFERENCE}"])]
    chosen = max(eligible, key=lambda arm: comparisons[f"{arm} - {REFERENCE}"]["j"], default=None)
    return dict(chosen=chosen, orders=SCREEN, comparisons=comparisons, timing=timing)


def held(root, data):
    screen = json.loads((root / "selection.json").read_text())
    arm = screen["chosen"]
    if arm is None:
        return dict(chosen=None, retained=False, comparisons={})
    assert arm in ARMS[1:] and tuple(screen["orders"]) == SCREEN
    cells = verified_cells(root, "held", (REFERENCE, arm), HELD, data)
    comparisons = contrasts(cells, ((arm, REFERENCE),), HELD)
    score = comparisons[f"{arm} - {REFERENCE}"]
    passed = all(score[k]["mean"] >= MIN_GAIN and score[k]["ci80"][0] > 0 for k in ("new", "returning"))
    return dict(chosen=arm, retained=passed, orders=HELD, comparisons=comparisons)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("select", "held"))
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    a = p.parse_args()
    data = V.D.load(a.data)
    result = (select if a.phase == "select" else held)(a.run, data)
    out = a.run / ("selection.json" if a.phase == "select" else "confirmation.json")
    with out.open("x") as stream:
        json.dump(result, stream, indent=1)
        stream.write("\n")
    print_contrasts(result["comparisons"])
    print("Frozen choice:", result["chosen"])
    if "retained" in result:
        print("Held-out retention threshold met:", result["retained"])


if __name__ == "__main__":
    main()
