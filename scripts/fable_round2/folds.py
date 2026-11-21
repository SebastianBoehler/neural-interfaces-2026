"""Registered Round 2 folds and pinned hard/nine25 Lane TB tapes."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "cleanslate/harness"), str(ROOT / "scripts/laneTB")]
import data as D

ORDERS = tuple(range(71124, 71132))
FOLDS = tuple(f"new_{p}" for p in D.PEOPLE) + tuple(f"ret_{p}" for p in D.RETURNING)
PARENT = Path("/leonardo_work/AIFAC_P02_1636/ni26/cap250/runs")
TAPES = PARENT / "laneTB-verify-21nov/out/hard"
LEAD = PARENT / "final-hard-teamb-21nov/zips/final_stack_hard_safe_st_team_b_nine25_tbgraph.zip"
LEAD_SHA = "6e09953709b2338507deaac1ec5f6afa4a21eef52de7814f7c2ae77f42e86ee9"
DATA_SHA = "2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327"
RIGHT = ("F6", "AF4", "FC6")
LEFT = ("F5", "AF3", "FC5")


def eye_mean(data):
    ix = {c: i for i, c in enumerate(data["ch_names"])}
    means = data["X"].mean(axis=-1, dtype=np.float64)
    return (means[:, [ix[c] for c in RIGHT]].sum(1) - means[:, [ix[c] for c in LEFT]].sum(1)) / 3


def streams(data, fold):
    mode, person = fold.split("_")
    sessions = (sorted(set(data["session"][data["person"] == person])) if mode == "new" else ("2", "3"))
    return [f"{person}|{session}" for session in sessions]


def pinned(fold, arm, order, stream, rows):
    session = stream.split("|")[1]
    directory = TAPES / fold / arm
    key = f"{order}_{session}"
    matches = []
    for path in sorted(directory.glob(f"pred_{fold}_*.npz")):
        with np.load(path, allow_pickle=False) as tape:
            if f"rows_{key}" not in tape:
                continue
            np.testing.assert_array_equal(tape[f"rows_{key}"], rows)
            y = tape[f"labels_{key}"]
            assert y.shape == rows.shape and np.isin(y, (0, 1, 2)).all()
            matches.append((y, str(tape["weights_sha"]), str(tape["source_sha"])))
    if len(matches) != 1:
        raise ValueError(f"expected one tape for {fold}/{arm}/{key}, found {len(matches)}")
    return matches[0]


def pinned_pair(fold, order, stream, rows):
    base, bh, bs = pinned(fold, "base", order, stream, rows)
    lead, lh, ls = pinned(fold, "nine25", order, stream, rows)
    assert bh == lh and bs == ls and len(bh) == len(bs) == 64
    return base, lead, bh


def standardise(values):
    q25, mid, q75 = np.percentile(values, (25, 50, 75))
    return np.clip((values - mid) / max(q75 - q25, 1e-9), -3, 3)
