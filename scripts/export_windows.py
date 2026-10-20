"""Export every official PROTEUS window with its metadata for fast experiments.

Windows come from the unchanged official NeuralBench pipeline (41 channels,
120 Hz, 4 s, microvolts). All three official splits are written to one file,
each window tagged with its official split, participant, session, interface
(the scoring context), BIDS run and onset, in official acquisition order.

``python scripts/export_windows.py START`` instead writes training crops
starting START s after the cue (official: 1.0; the task lasts until 8 s)
to data/proteus_windows_startSTART.npz, same pipeline and window order.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "2026-competition"
sys.path.insert(0, str(REPO))

from benchmark_utils.nb_task import load_task  # noqa: E402
from tracks.bci_decoding.datasets.bci_studies import (  # noqa: E402
    _proteus_interface,
    _proteus_run,
)

OUT = ROOT / "data/proteus_windows.npz"


def main():
    start = float(sys.argv[1]) if len(sys.argv) > 1 else None
    out = OUT if start is None else OUT.with_name(f"proteus_windows_start{start:g}.npz")
    loaders, meta = load_task(
        "eeg", "_motor_imagery_stream",
        data_dir=ROOT / "data/neural_compet",
        device="cpu", batch_size=64, seed=42, num_workers=0,
        target_transform=lambda y: y.argmax(-1),
        context_of=_proteus_interface, order_of=_proteus_run,
        overrides=None if start is None else {"start": start},
    )
    cols = {k: [] for k in ("X", "y", "split", "subject", "session",
                            "interface", "task", "run", "onset")}
    for split, loader in loaders.items():
        ds = loader.dataset
        segments = ds.seg_ds.segments
        for pos in range(len(ds)):
            X, y, info = ds[pos]
            trig = segments[int(ds.index[pos])].trigger.to_dict()
            cols["X"].append(X.numpy())
            cols["y"].append(int(y))
            cols["split"].append(split)
            cols["subject"].append(str(trig["subject"]))
            cols["session"].append(str(trig["session"]))
            cols["interface"].append(_proteus_interface(trig))
            cols["task"].append(str(trig["task"]))
            cols["run"].append(_proteus_run(trig))
            cols["onset"].append(float(info["onset"]))
        print(split, len(ds), flush=True)
    arrays = {k: np.asarray(v) for k, v in cols.items()}
    arrays["X"] = arrays["X"].astype(np.float32)
    assert arrays["X"].shape[1:] == (41, 480)
    assert np.isfinite(arrays["X"]).all()
    np.savez(out, ch_names=np.asarray(meta["ch_names"]), **arrays)
    summary = {
        "n_windows": int(len(arrays["y"])),
        "cells": sorted({f"{s}|{se}|{i}|{sp}" for s, se, i, sp in zip(
            arrays["subject"], arrays["session"], arrays["interface"],
            arrays["split"])}),
    }
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
