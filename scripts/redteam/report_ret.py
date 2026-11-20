"""Generate strict paired RT-1 score tables from collected Leonardo tapes.

Usage: report_ret.py DATA.npz RECEIPT_DIR
Each arm directory contains pred_ret_P04.npz and pred_ret_P10.npz. Missing
arms are omitted so partial Slurm results can be reported without pooling them.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from score_ret import PEOPLE, cells

ORDERS = tuple(range(71000, 71004))
VISITS = ("2", "3")
FACES = ("BrainHero", "Graz")
DOSES = ("g6", "g12", "beta02", "beta04", "kappa40")
ARMS = ("hardcb", "hardcb_safe", "R0z", "R1a", "R5", "R2", "R4", *DOSES, "leak_all", "noroute")
DRAWS = 20_000
SEED = 20261121


def pinned_rows(data, orders, visits, people):
    stream = np.char.add(np.char.add(data["person"], "|"), data["session"])
    source_rows = {s: np.flatnonzero(stream == s) for s in dict.fromkeys(stream)}
    plans = {}
    for order in orders:
        rng = np.random.default_rng([order, 0, 0])
        shuffled = {s: rows[rng.permutation(len(rows))] for s, rows in source_rows.items()}
        for person in people:
            for visit in visits:
                plans[person, order, visit] = shuffled[f"{person}|{visit}"]
    return plans


def read_tape(data, path, who, plans, orders=ORDERS, visits=VISITS):
    with np.load(path, allow_pickle=False) as tape:
        expected = {f"{order}_{visit}" for order in orders for visit in visits}
        for prefix in ("rows_", "labels_"):
            if {key[len(prefix):] for key in tape.files if key.startswith(prefix)} != expected:
                raise ValueError(f"{path}: incomplete or unexpected {prefix} coverage")
        for order in orders:
            for visit in visits:
                key = f"{order}_{visit}"
                rows, labels = tape[f"rows_{key}"], tape[f"labels_{key}"]
                np.testing.assert_array_equal(rows, plans[who, order, visit], err_msg=f"{path}:{key}")
                if rows.shape != labels.shape or labels.dtype.kind not in "iu" or not np.isin(labels, (0, 1, 2)).all():
                    raise ValueError(f"{path}: invalid labels in {key}")
                for face in FACES:
                    if set(data["y"][rows][data["interface"][rows] == face]) != {0, 1, 2}:
                        raise ValueError(f"{path}: incomplete class coverage in {key}, {face}")
        table = cells(data, tape)
        values = np.array([[table[order, visit, face][0] for visit in visits for face in FACES]
                           for order in orders])
        diagnostics = {key: tape[key] for key in tape.files if key.startswith(("routed_", "labels_"))}
    return values, diagnostics


def interval(gains):
    lo, hi = np.quantile(gains, (0.1, 0.9))
    return f"[{lo:+.2f}, {hi:+.2f}]"


def paired(gain):
    rng = np.random.default_rng(SEED)
    p, o = gain.shape
    per_person = gain[:, rng.integers(o, size=(DRAWS, o))].mean(axis=2)
    people = rng.integers(p, size=(DRAWS, p, 1))
    orders = rng.integers(o, size=(DRAWS, p, o))
    nested = gain[people, orders].mean(axis=(1, 2))
    return [interval(a) for a in per_person], interval(nested)


def routed_changes(arm, reference):
    counts = []
    for pred, base in zip(arm, reference, strict=True):
        count = 0
        for key in (f"{o}_{v}" for o in ORDERS for v in VISITS):
            routed = pred[f"routed_{key}"]
            if routed.dtype != np.bool_ or routed.shape != pred[f"labels_{key}"].shape:
                raise ValueError(f"invalid route trace at {key}")
            count += np.count_nonzero(routed & (pred[f"labels_{key}"] != base[f"labels_{key}"]))
        counts.append(count)
    return counts


def main(data_path, receipts):
    with np.load(data_path, allow_pickle=False) as source:
        data = {key: source[key] for key in ("y", "interface", "subject", "session")}
    data["person"] = np.array([s.split("/")[-1] for s in data["subject"]])
    plans = pinned_rows(data, ORDERS, VISITS, PEOPLE)
    tables, traces = {}, {}
    for arm in ("stock", *ARMS):
        files = [receipts / arm / f"pred_ret_{who}.npz" for who in PEOPLE]
        if not all(f.is_file() for f in files):
            print(f"Missing {arm}: " + ", ".join(f.name for f in files if not f.is_file()), file=sys.stderr)
            continue
        values, diagnostics = zip(*(read_tape(data, f, who, plans) for f, who in zip(files, PEOPLE, strict=True)))
        tables[arm], traces[arm] = np.stack(values), diagnostics

    print("| Arm | Primary reference | P04 BA | P10 BA | P04 Δ [80%] | P10 Δ [80%] | Equal cells Δ [80%] | 20/80 Δ* |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    print(f"| stock | — | {tables['stock'][0].mean():.2f} | {tables['stock'][1].mean():.2f} | — | — | — | — |")
    order_lines = []
    for arm in (a for a in ARMS if a in tables):
        ref = "stock" if arm in ("hardcb", "hardcb_safe", "leak_all", "noroute") else "hardcb"
        gain = (tables[arm] - tables[ref]).mean(axis=2)
        means = gain.mean(axis=1)
        person_ci, nested_ci = paired(gain)
        print(f"| {arm} | {ref} | {tables[arm][0].mean():.2f} | {tables[arm][1].mean():.2f} | "
              f"{means[0]:+.2f} {person_ci[0]} | {means[1]:+.2f} {person_ci[1]} | "
              f"{gain.mean():+.2f} {nested_ci} | {means @ [0.2, 0.8]:+.2f} |")
        order_lines.append("| " + " | ".join([arm, *("/".join(f"{d:+.2f}" for d in row) for row in gain)]) + " |")
    print("\n| Arm | P04 Δ by order | P10 Δ by order |\n|---|---|---|")
    print("\n".join(order_lines))
    print("\n| Arm | Δ P04 / P10 vs stock | Δ P04 / P10 vs hardcb_safe | Routed labels changed P04 / P10 |")
    print("|---|---|---|---|")
    for arm in (a for a in ARMS if a in tables):
        means = lambda ref: " / ".join(f"{n:+.2f}" for n in (tables[arm] - tables[ref]).mean(axis=(1, 2)))
        checks = " / ".join(map(str, routed_changes(traces[arm], traces["hardcb"]))) if arm in DOSES else "—"
        print(f"| {arm} | {means('stock')} | {means('hardcb_safe') if 'hardcb_safe' in tables else 'pending'} | {checks} |")

    in_sample = receipts / "insample_P04.npz"
    if in_sample.exists():
        plan = pinned_rows(data, (71000,), ("1",), ("P04",))
        scores, _ = read_tape(data, in_sample, "P04", plan, (71000,), ("1",))
        print(f"\nH3 P04 visit 1 (71000): {scores.mean():.2f}% cell mean "
              f"(BrainHero {scores[0, 0]:.2f}%, Graz {scores[0, 1]:.2f}%).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("receipts", type=Path)
    args = parser.parse_args()
    main(args.data, args.receipts)
