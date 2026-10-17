"""Check real PROTEUS splits and tensors before baseline training."""

import json
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "2026-competition"
sys.path.insert(0, str(REPO))

from benchmark_utils.nb_task import load_task  # noqa: E402
from tracks.bci_decoding.datasets.bci_studies import (  # noqa: E402
    _proteus_interface,
    _proteus_run,
)


def main():
    loaders, meta = load_task(
        "eeg", "_motor_imagery_stream",
        data_dir=ROOT / "data/neural_compet",
        device="cpu", batch_size=64, seed=42, num_workers=0,
        target_transform=lambda y: y.argmax(-1),
        context_of=_proteus_interface, order_of=_proteus_run,
    )
    assert (meta["n_chans"], meta["n_times"], meta["sfreq"]) == (41, 480, 120)
    summary = {"metadata": meta, "splits": {}}
    participants = {}
    for name, loader in loaders.items():
        windows = loader.dataset
        subjects = sorted({str(seg.trigger.to_dict()["subject"])
                           for seg in windows.seg_ds.segments})
        participants[name] = set(subjects)
        labels = Counter()
        n_windows = 0
        for X, y, info in loader:
            assert X.ndim == 3 and X.shape[1:] == (41, 480), X.shape
            assert X.dtype == torch.float32
            assert torch.isfinite(X).all(), f"Non-finite data in {name}"
            assert y.ndim == 1 and y.dtype == torch.int64
            assert ((y >= 0) & (y < 3)).all(), name
            if name == "test":
                assert len(X) == 1, "Evaluation must stream one window at a time"
                assert "stream_id" in info and "context_id" in info
            labels.update(y.tolist())
            n_windows += len(X)
        assert n_windows == len(windows), name
        assert set(labels) == {0, 1, 2}, labels
        summary["splits"][name] = {
            "participants": subjects, "n_windows": n_windows,
            "class_counts": dict(sorted(labels.items())),
        }
    assert len(participants["train"]) == 6
    assert len(participants["val"]) == 2
    assert len(participants["test"]) == 2
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        assert participants[left].isdisjoint(participants[right]), (left, right)
    summary["participant_overlap"] = False
    path = ROOT / "runs/baseline/data_audit.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
