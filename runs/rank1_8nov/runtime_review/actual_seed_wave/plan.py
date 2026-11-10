"""Fixed participant separation and nested solver fits for the actual runner gate."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments import protocol

EVAL = ("P04", "P09", "P10")
LATE = ("P04", "P10")
SINGLES = ("a", "b", "c")
PAIRS = ("ab", "ac", "bc")
BUILDS = (*SINGLES, *PAIRS, "abc")


def metadata():
    with np.load(protocol.DATA, allow_pickle=False) as z:
        meta = {key: z[key] for key in ("subject", "session", "interface", "split")}
    meta["stream"] = np.char.add(np.char.add(meta["subject"], "|"), meta["session"])
    return meta


def partition(meta):
    person = np.array([name.rsplit("/", 1)[-1] for name in meta["subject"]])
    evaluation = np.isin(person, EVAL)
    if set(person[meta["split"] == "test"]) != set(EVAL) - {"P10"}:
        raise ValueError("Unexpected official public test participants")
    source_people = np.unique(person[~evaluation])
    assert len(source_people) == 7 and set(person[evaluation]) == set(EVAL)
    groups = {k: np.flatnonzero(np.isin(person, source_people[i::3])) for i, k in enumerate(SINGLES)}
    source = np.flatnonzero(~evaluation)
    np.testing.assert_array_equal(np.sort(np.concatenate(list(groups.values()))), source)
    target = np.flatnonzero(evaluation)
    later = np.flatnonzero(evaluation & np.isin(person, LATE) & (meta["session"] != "1"))
    assert set(person[later]) == set(LATE) and set(meta["interface"][later]) == {"Graz", "BrainHero"}
    return groups, target, later


def source_rows(meta, key):
    if key not in BUILDS:
        raise ValueError(f"Unknown source build {key}")
    groups, target, _ = partition(meta)
    rows = np.sort(np.concatenate([groups[k] for k in key]))
    assert not np.intersect1d(meta["subject"][rows], meta["subject"][target]).size
    return rows


def slice_source(data, key):
    rows = source_rows(data, key)
    n = len(data["subject"])
    local = {k: v[rows] if isinstance(v, np.ndarray) and v.ndim and len(v) == n else v
             for k, v in data.items()}
    assert len(local["X"]) == len(local["y"]) == len(rows)
    assert not np.any(local["split"] == "test")
    assert not np.isin([x.rsplit("/", 1)[-1] for x in local["subject"]], EVAL).any()
    return rows, local
