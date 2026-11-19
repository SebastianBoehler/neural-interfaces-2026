"""Use actual pinned Scherer rows and original RTX archives without gold labels."""
import numpy as np

from data import permutation
from scope import external, known
from settings import stamp

SEEDS = (22411, 23411)
DATA_SHA = "a6188799753276451235678d08e92fd79162a95ec8099dbbb85e2ed9e2701e91"
ORDERS = {
    0: "63ef8f828ff49cfa2b2a9152571b5a496810d1d7834b2f38e2be1c345307c5d0",
    22411: "e334c9e410b0b7826613823032d42166dc2ccf2686d756660ceec313810815ed",
    23411: "c21aeee3614815e954431f600269fb29d8706cf4c47c325b6dd5352224a682ef",
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

