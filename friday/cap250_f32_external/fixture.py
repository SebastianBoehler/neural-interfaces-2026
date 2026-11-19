"""Use actual pinned Scherer rows and original RTX archives without gold labels."""
import json
from pathlib import Path

import numpy as np

from data import permutation
from scope import external, known
from settings import stamp

SEEDS = (14411, 15411)
ARMS = ("stock", "hard", "joint")
EXPORTS = ("hard", "detached", "joint")
DATA_SHA = "a6188799753276451235678d08e92fd79162a95ec8099dbbb85e2ed9e2701e91"
ORDERS = {
    0: "63ef8f828ff49cfa2b2a9152571b5a496810d1d7834b2f38e2be1c345307c5d0",
    14411: "2a4bd8fc50262c014d245966290daafe3663727f729924255f7042220604863e",
    15411: "3d6d7243eafc6a0c0cd8b0d4cfcd4000f6830b97f7a09f1be7980c0da8287699",
}


def schedule(subject, day):
    stream = np.char.add(np.char.add(subject, "|"), day)
    assert stream.shape == (2130,) and len(set(subject)) == 9 and len(set(stream)) == 18
    plans = {}
    for seed, digest in ORDERS.items():
        p = permutation(stream, None, "full", seed).astype("<i8")
        assert stamp(p) == digest
        plans[seed] = {s: p[stream[p] == s] for s in sorted(set(stream))}
    return plans


def chosen(mode, index):
    eeg, who, channels, earlier, anchor, archive = external(index)
    canonical = earlier[0][who+"|1"]
    assert canonical.tolist() == anchor["rows"]
    sessions = {}
    if mode == "smoke":
        assert who == "Scherer2015/G"
        x, rows, classes = known(channels)
        sessions[f"{who}|1|0"] = (eeg, canonical)
        sessions["P10-3"] = (x, rows)
        expected = {"P10-3": classes, f"{who}|1|0": anchor["labels"]}
    else:
        with np.load("/external.npz", allow_pickle=False) as z:
            plans = schedule(z["subject"], z["session"])
        assert np.array_equal(plans[0][who+"|1"], canonical)
        for seed in SEEDS:
            for day in (1, 2):
                key = f"{who}|{day}"
                sessions[f"{key}|{seed}"] = (eeg, plans[seed][key])
        expected = {}
    return sessions, (eeg, canonical, anchor, archive), channels, who, expected


def actor_lock(path):
    lock = json.loads(Path(path).read_text())
    if lock["f32e_source_only"] is not True or set(lock["actors"]) != set(EXPORTS):
        raise ValueError("actor lock must contain all three exact all-public-source exports")
    if any(len(lock["actors"][kind]) != 2 for kind in EXPORTS):
        raise ValueError("original fixed seed pair missing")
    return lock
