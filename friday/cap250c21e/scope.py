"""The pinned RTX fixture and two previously unused external deliveries."""

import json
from pathlib import Path

import numpy as np
import torch

import replay
from data import permutation
from settings import sha, stamp

SEEDS = (10411, 11411)
ORDERS = {
    0: "63ef8f828ff49cfa2b2a9152571b5a496810d1d7834b2f38e2be1c345307c5d0",
    10411: "ff77624e04d2ccf0f7df1d3ad44fb26917096f6389da5e826d2820efe1414338",
    11411: "ada2aadf7b3ed3b428b70a8fc50aab6009fea3b28d04105bcda0590dd1c0c74c",
}
ARMS = ("stock", "fbl2")


def schedule(subject, day):
    streams = np.char.add(np.char.add(subject, "|"), day)
    assert streams.shape == (2130,) and len(set(subject)) == 9 and len(set(streams)) == 18
    plans = {}
    for s, digest in ORDERS.items():
        p = permutation(streams, None, "full", s).astype("<i8")
        assert stamp(p) == digest
        plans[s] = {k:p[streams[p] == k] for k in sorted(set(streams))}
    return plans


def external(index):
    with np.load("/external.npz", allow_pickle=False) as z:
        eeg = torch.from_numpy(z["X"])
        subject, day, channels = z["subject"], z["session"], z["ch_names"].tolist()
    assert eeg.shape == (2130, 41, 480) and eeg.dtype == torch.float32
    who = sorted(set(subject))[index]
    archive = Path("/rtx_pair" if index in (0,5) else "/rtx_other") / f"rtx_cap_{index}.json"
    record = json.loads(archive.read_text())
    assert record["person"] == index and record["arm"] == "cap" and "RTX 2080 Ti" in record["device"]
    order = schedule(subject, day)
    anchor = record["sessions"][f"{who}|1|0"]
    assert order[0][who+"|1"].tolist() == anchor["rows"]
    return eeg, who, channels, order, anchor, sha(archive)


def known(channels):
    x, streams, names = replay.read_source()
    assert names == channels
    rows = dict(streams)["P10-3"][:31]
    e = json.loads(Path("/known_rtx.json").read_text())["shared"]["recognized"]
    a = json.loads(Path("/known.json").read_text())["streams"]["first"]["P10-3"]
    assert rows.tolist() == e["rows"] and e["activity"]["recognized"] == a["activity"]["recognized"] == 10
    return x, rows, a["labels"][:31]
