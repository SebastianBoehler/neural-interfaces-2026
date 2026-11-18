"""Bind the exact same-platform F17 flat and jitter controls."""
import json
from pathlib import Path

import numpy as np

from anchors import DATA_SHA, ORIGIN_SHA, RUNTIME, row_sha, schedule, sha
from perturb import random_tape


HERE = Path(__file__).resolve().parent
KINDS = ("hard", "jitter", "hierarchy", "hierarchy_jitter")
MODES = ("ordinary", "bounded")
ARMS = tuple(f"{k}_{m}" for k in KINDS for m in MODES)
F17_ACTOR = "7ed98f96ba6b05165a1bc035a20af0d7dc516b9b1d61c82533abcc85ecbf2372"
F17_BANK = "77a25afbba7061dde2c781f3e8fb810b49d653aa57d09dcbc4b5abc0bb2abb37"
F17_PINS = "ea387a43cbff8a6f91e3c86cd9b1b06c5d76eff49c3e7256a99fbaf805d44c6f"


def controls(root, folder, held, seed, smoke, rows, stream, people):
    pins = HERE / "f17_pins.json"
    if sha(root / "actor.sha256") != F17_ACTOR or sha(pins) != F17_PINS:
        raise ValueError("authentic F17 exact controls changed")
    b, trace = schedule(rows, stream, seed, 4 if smoke else 40)
    signs, direction, masks, tape = random_tape(b, len(rows), held, seed)
    count = sum(map(len, masks))
    tag = f"{'smoke_' if smoke else ''}{seed}_{held}"
    index = json.loads(pins.read_text())
    fixed = dict(held=held, seed=seed, epochs=len(b), dataset=DATA_SHA, origin=ORIGIN_SHA, runtime=RUNTIME,
                 source_rows=row_sha(rows), source_people=people, actor=F17_ACTOR, batches=trace, tape=tape,
                 q_gradients_nonzero=0, held_labels_opened=False, parameters=347903,
                 train_rows_seen=count, planned_rows=count // 4, steps=len(masks))
    reference = {}
    for kind in ("hard", "jitter"):
        name = f"{kind}_{tag}"
        p = folder / f"{name}.pt"
        pin = index[name]
        meta = json.loads(p.with_suffix(".json").read_text())
        expected = dict(fixed, kind=kind, checkpoint=pin["checkpoint"], state=pin["state"],
                        perturbed_rows=count // 4 if kind == "jitter" else 0)
        if (sha(p) != pin["checkpoint"] or sha(p.with_suffix(".json")) != pin["receipt"] or any(
                meta[k] != v for k, v in expected.items()) or "H100" not in meta["device"] or not np.isfinite(meta["scale"])):
            raise ValueError("source control, held exclusion or F17 exact tape differs")
        reference[kind] = meta
    if any(reference["hard"][k] != reference["jitter"][k] for k in ("initial", "scale", "device", "torch", "braindecode")):
        raise ValueError("original controls were not paired")
    return reference, b, direction, masks, tape
