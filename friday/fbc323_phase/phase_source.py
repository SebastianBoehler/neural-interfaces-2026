"""Pin the genuine F16 raw-only controls and original source schedule."""
import json
from pathlib import Path

import numpy as np

from anchors import DATA_SHA, ORIGIN_SHA, RUNTIME, row_sha, schedule, sha


HERE = Path(__file__).resolve().parent
KINDS = ("hard", "jitter", "phase")
MODES = ("ordinary", "bounded")
ARMS = tuple(f"{k}_{m}" for k in KINDS for m in MODES)
F16_ACTOR = "33d13bad6fe04b0ef14d8fb4f91f5034cf56a8d928ed38b3a09eca82a4c8c0d5"
F16_BANK = "299ba53d78116b2e015654cb90f0d43d1129bf11a0ad083a97a964fe60c41a0e"
F16_PINS = "96100b45f9c3602b3c4d76a77b6143e8e959a7954859e52cb604c43bc8b78663"


def authenticated(f16, folder, held, seed, smoke, rows, stream, people):
    if sha(f16 / "actor.sha256") != F16_ACTOR or sha(HERE / "f16_pins.json") != F16_PINS:
        raise ValueError("the authenticated F16 source ancestor differs")
    tag = f"{'smoke_' if smoke else ''}{seed}_{held}"
    path = folder / f"hard_{tag}.pt"
    receipt = path.with_suffix(".json")
    pin = json.loads((HERE / "f16_pins.json").read_text())[path.stem]
    got = json.loads(receipt.read_text())
    b, trace = schedule(rows, stream, seed, 4 if smoke else 40)
    expected = dict(kind="hard", held=held, seed=seed, epochs=len(b), dataset=DATA_SHA,
                    origin=ORIGIN_SHA, runtime=RUNTIME, source_rows=row_sha(rows), source_people=people,
                    actor=F16_ACTOR, batches=trace, state=pin["state"], checkpoint=pin["checkpoint"],
                    q_gradients_nonzero=0, held_labels_opened=False, parameters=347903, steps=sum(map(len, b)))
    if (sha(path) != pin["checkpoint"] or sha(receipt) != pin["receipt"] or
            any(got[k] != v for k, v in expected.items()) or not np.isfinite(got["scale"]) or not got["scale"] > 0):
        raise ValueError("pinned F16 hard-CE source checkpoint differs")
    return got, b
