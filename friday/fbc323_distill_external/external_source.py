"""Pin F15e's two independently authenticated all-source teachers."""
import json
from pathlib import Path

import numpy as np

from anchors import DATA_SHA, ORIGIN_SHA, RUNTIME, row_sha, schedule, sha


HERE = Path(__file__).resolve().parent
F15E_ACTOR = "9d552ab6df96d7aaa171cb523b2a90ba9988f2f698e1544c7802bb960d9ea0e0"
F15E_BANK = "ff25cdfee588ad9085ca85d9ac97bf26218621aca5a0b454a9813e2caaff2884"
F16_ACTOR = "33d13bad6fe04b0ef14d8fb4f91f5034cf56a8d928ed38b3a09eca82a4c8c0d5"
TEACHERS = (
    ("0a2c4063a23e197040933db66120ac311aa7f25f8a64bbe9a41923fbc7d78a59",
     "be42ef9c625759e6c059a34640b918949a65e4d994cabcf52d00cd2abdc7add3",
     "439203dc0d6aaf305d2b37d4fd6c77db5a764fdf7427684735c6e6a2a7150058"),
    ("2673e4d552d7255d9e96db324c328badfefe7a89f02d3ffcb065bbcde25f8641",
     "f059f5c8882ebb102364ca9da73b16744a262ea9556cab6b5ec1448a8acb60fe",
     "d22082a4c23a720e6dc2d42282a07f14e7fd5df57a20760be667753e02a9b5b0"),
)


def all_source(path, waveforms):
    if sha(path) != DATA_SHA:
        raise ValueError("raw PROTEUS source identity differs")
    with np.load(path, allow_pickle=False) as z:
        subject, day = z["subject"], z["session"]
        x = z["X"] if waveforms else None
    people = sorted(set(subject))
    if people != [f"Dreyer2026Proteus/P{i:02d}" for i in range(1, 11)] or len(subject) != 2655:
        raise ValueError("all-source roster differs from F15e")
    if waveforms and (x.dtype != np.float32 or x.shape != (2655, 41, 480) or not np.isfinite(x).all()):
        raise ValueError("fixed raw PROTEUS input differs")
    return x, np.arange(len(subject)), np.char.add(np.char.add(subject, "|"), day), people


def teacher_receipt(root, directory, rows, stream, people, seed):
    if sha(root / "actor.sha256") != F15E_ACTOR:
        raise ValueError("F15e training actor differs")
    p = directory / f"dropout_{seed}_all_source.pt"
    checkpoint, receipt, state = TEACHERS[seed]
    if sha(p) != checkpoint or sha(p.with_suffix(".json")) != receipt:
        raise ValueError("F15e teacher differs from its sealed external receipt")
    meta = json.loads(p.with_suffix(".json").read_text())
    batches, trace = schedule(rows, stream, seed, 40)
    expected = dict(seed=seed, kind="dropout", held="all_source", epochs=40, steps=sum(map(len, batches)),
                    state=state, actor=F15E_ACTOR, runtime=RUNTIME, origin=ORIGIN_SHA, dataset=DATA_SHA,
                    source_rows=row_sha(rows), source_people=people, batches=trace, checkpoint=checkpoint,
                    held_labels_opened=False, parameters=347903)
    if (any(meta[k] != v for k, v in expected.items()) or not np.isfinite(meta["scale"]) or not meta["scale"] > 0
            or 4 * meta["active_mask_rows"] != meta["train_rows_seen"]):
        raise ValueError("F15e frozen teacher source identity, scale or schedule differs")
    return p, meta
