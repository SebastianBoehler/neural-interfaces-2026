"""Recompute both geometric source policies in every fold before any labels."""
import json

import numpy as np

from anchors import PEOPLE, schedule, split
from replay_f13 import DATA, DATA_SHA, sha
from withholding import plan


def run():
    if sha(DATA) != DATA_SHA:
        raise ValueError("official waveform input differs")
    with np.load(DATA, allow_pickle=False) as z:
        x, channels, subject, day = (z[k] for k in ("X", "ch_names", "subject", "session"))
    receipts = {}
    frequency = np.zeros((2, 41), dtype=np.int64)
    for held in PEOPLE:
        rows, streams, _ = split(subject, day, held)
        for seed in (0, 1):
            batches, _ = schedule(rows, streams, seed, 40)
            local, dispersed, _, tape, values = plan(x[rows], channels, batches, held, seed)
            if held == "P01" and seed == 0:
                smoke, _ = schedule(rows, streams, seed, 4)
                _, _, _, first, _ = plan(x[rows], channels, smoke, held, seed)
                if first != "b902590eb5bd27ad7cdd1fc5f419b37f99dbc979045b559de44375b610486b58":
                    raise ValueError("Galvani smoke tape did not reproduce the actual Ferranti source layout")
            for i, group in enumerate((local, dispersed)):
                frequency[i] += np.bincount(group.reshape(-1), minlength=41)
            receipts[f"{held}_{seed}"] = dict(tape=tape, source_rows=len(rows), **values)
    print(json.dumps(dict(source_label_free=True, receipts=receipts, sensor_names=channels.tolist(),
                          channel_mask_counts=frequency.tolist()), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    run()
