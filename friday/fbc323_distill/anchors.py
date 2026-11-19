"""Read pinned F15 controls and build the original label-free source schedule."""
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DATA_SHA = "2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327"
ORIGIN_SHA = "88437e69b5eaf7ff84a3635df88fe68b38394abbc527629b0f41c7fcc24eb6d5"
RUNTIME = "da21360c9a85d7561eee8f0f68367b545797641a69e3726354275cbf404ec30e"
F15_ACTOR = "04ec8dd8e7593088a1b882e8fcf75cf3ac9bf67ff737c14c1903b70f97ea1be3"
PIN_SHA = "e04d4fae1b8a4838d199ca6f85e75b2d6ecd09749558a476f91f8b13b1a1b223"
PEOPLE = tuple(f"P{i:02d}" for i in range(1, 11))
KINDS = ("hard", "trial", "deranged")
MODES = ("ordinary", "bounded")
ARMS = tuple(f"{kind}_{mode}" for kind in KINDS for mode in MODES)


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def row_sha(rows):
    return hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest()


def split(subject, visit, held):
    person = np.asarray([s.rsplit("/", 1)[-1] for s in subject])
    rows = np.flatnonzero(person != held)
    stream = np.char.add(np.char.add(subject, "|"), visit)
    roster = sorted(set(subject[rows]))
    if (len(subject) != 2655 or set(person) != set(PEOPLE) or set(person[rows]) != set(PEOPLE) - {held}
            or len(roster) != 9 or any(not p.startswith("Dreyer2026Proteus/") for p in roster)):
        raise ValueError("source/held split does not match the original nine-person fit")
    return rows, stream[rows], roster


def schedule(rows, stream, seed, epochs):
    import model322

    model322.SEED = seed
    batches = model322.batches(stream, epochs)
    trace = np.concatenate([np.concatenate(epoch) for epoch in batches])
    if any(not np.array_equal(np.unique(trace_epoch), np.arange(len(rows)))
           for trace_epoch in (np.concatenate(epoch) for epoch in batches)):
        raise ValueError("an epoch misses canonical source rows")
    return batches, row_sha(rows[trace])


def anchor(directory, kind, held, seed, smoke, rows, stream, roster):
    index = HERE / "f15_pins.json"
    if sha(index) != PIN_SHA:
        raise ValueError("fixed F15 source pins differ")
    tag = f"{'smoke_' if smoke else ''}{kind}_{seed}_{held}"
    path = directory / f"{tag}.pt"
    receipt = path.with_suffix(".json")
    pin = json.loads(index.read_text())[tag]
    if sha(path) != pin["checkpoint"] or sha(receipt) != pin["receipt"]:
        raise ValueError(f"F15 sealed parent checkpoint differs for {tag}")
    meta = json.loads(receipt.read_text())
    epochs = 4 if smoke else 40
    batches, trace = schedule(rows, stream, seed, epochs)
    expected = dict(kind=kind, held=held, seed=seed, epochs=epochs, dataset=DATA_SHA, origin=ORIGIN_SHA,
                    runtime=RUNTIME, actor=F15_ACTOR, source_rows=row_sha(rows), source_people=roster, batches=trace,
                    held_labels_opened=False, parameters=347903, steps=sum(map(len, batches)),
                    checkpoint=pin["checkpoint"], state=pin["state"])
    if any(meta[k] != v for k, v in expected.items()) or not np.isfinite(meta["scale"]) or meta["scale"] <= 0:
        raise ValueError(f"F15 checkpoint controls differ for {tag}")
    return path, meta
