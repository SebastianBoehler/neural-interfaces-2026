"""Compare R1a against original Lane R4 and clean H1 tapes on their recorded rows."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

from check_r1a import load_templates, MODES, replay, verify

import numpy as np
import torch


def replay_tape(template, windows, selected, expected, tag, smoke):
    labels, recognised, late, seconds = replay(template, windows[selected].to("cuda:0"))
    differences = int(np.count_nonzero(labels != expected))
    if smoke:
        if recognised >= 0 or late:
            raise ValueError(f"{tag}: unexpected recognition or adaptation in the first eight windows")
    elif not 0 <= recognised < 100 or late != [100]:
        raise ValueError(f"{tag}: expected recognition and the one late refit, got {recognised}, {late}")
    result = dict(tape=tag, windows=len(labels), mismatches=differences,
                  recognition=recognised, late_refits=late, seconds=seconds)
    print(json.dumps(result), flush=True)
    return result


def read_tape(path, row_key, label_key, sessions, session, smoke):
    with np.load(path, allow_pickle=False) as tape:
        rows, labels = tape[row_key], tape[label_key]
    np.testing.assert_array_equal(np.sort(rows), np.flatnonzero(sessions == session))
    if labels.shape != rows.shape or not np.isin(labels, (0, 1, 2)).all():
        raise ValueError(f"{path}: invalid frozen labels")
    return (rows[:8], labels[:8]) if smoke else (rows, labels)


def run(a):
    verify(a.zips, a.mode)
    with np.load(a.data, allow_pickle=False) as data:
        sessions = np.char.add(np.char.add(data["subject"], "|"), data["session"])
        channels, windows = data["ch_names"], torch.from_numpy(data["X"])
    exact, matched = [], []
    people = ("P04",) if a.smoke else ("P04", "P10")
    visits = ("2",) if a.smoke else ("2", "3")
    orders = (71000,) if a.smoke else range(71000, 71004)
    with ExitStack() as stack:
        _, final = load_templates(stack, a.zips, a.mode, channels, "cuda:0")
        for person in people:
            with ExitStack() as fold_stack:
                fold = a.fold_root / f"ret_{person}/pkg"
                _, fixture = load_templates(fold_stack, a.zips, a.mode, channels, "cuda:0", fold)
                for visit in visits:
                    session = f"Dreyer2026Proteus/{person}|{visit}"
                    for order in orders:
                        key = f"{order}_{visit}"
                        path = a.clean_tapes / f"pred_ret_{person}.npz"
                        rows, labels = read_tape(path, f"rows_{key}", f"labels_{key}", sessions, session, a.smoke)
                        tag = f"H1/{person}/{visit}/{order}"
                        matched.append(replay_tape(fixture, windows, rows, labels, tag, a.smoke))
                        exact.append(replay_tape(final, windows, rows, labels, tag, a.smoke))
                    for order in (("full0",) if a.smoke else ("full0", "full1", "within0", "within1")):
                        tag = f"{order}_{person}_{visit}"
                        path = a.historic_tapes / f"{tag}/{tag}.npz"
                        rows, labels = read_tape(path, "rows", "R1a|labels", sessions, session, a.smoke)
                        exact.append(replay_tape(final, windows, rows, labels, f"R4/{tag}", a.smoke))
    a.out.write_text(json.dumps(dict(exact=exact, fold=matched), indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=MODES)
    for name in ("zips", "data", "out", "fold_root", "clean_tapes", "historic_tapes"):
        p.add_argument(name, type=Path)
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())
