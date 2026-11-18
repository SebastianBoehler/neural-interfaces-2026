"""Label-blind real-EEG proof that recorded-order EA is not shuffled-delivery EA.

This checks AlignedNet's actual preprocessing with a source-only prior. It does
not estimate the error in a released model's probabilities or accuracy.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments"))
from deep import AlignedNet

PINS = {
    "data/proteus_windows.npz": "2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327",
    "runs/shuffle_perms.npz": "6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f",
}


def main():
    for name, expected in PINS.items():
        with (ROOT / name).open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
                raise ValueError(f"changed audit input: {name}")
    with np.load(ROOT / "data/proteus_windows.npz", allow_pickle=False) as data:
        people = data["subject"]
        sessions = data["session"]
        held = np.flatnonzero(people == "Dreyer2026Proteus/P01")
        held = held[sessions[held] == sessions[held[0]]]
        source = np.flatnonzero(people == "Dreyer2026Proteus/P02")[:24]
        eeg = data["X"][np.r_[held, source]]
    with np.load(ROOT / "runs/shuffle_perms.npz", allow_pickle=False) as data:
        order = data["full|0|all_loso"]
    delivered = order[np.isin(order, held)]
    if not np.array_equal(np.sort(delivered), held):
        raise ValueError("audit session order is not a bijection")
    first = int(np.searchsorted(held, delivered[0]))
    if first == 0:
        raise ValueError("pinned first delivery no longer exposes the recorded-prefix mismatch")
    model = AlignedNet({"X": eeg}, pretrained=False)
    model.G = model.C[len(held):].mean(0)
    model.scale = float(np.sqrt(np.trace(model.G) / len(model.G)))
    recorded = model._aligned(np.arange(len(held)), causal=True)[first]
    immediate = model._aligned(np.array([first]), causal=True)[0]
    streamed = model._aligned(np.searchsorted(held, delivered), causal=True)[0]
    np.testing.assert_array_equal(immediate, streamed)
    error = float(np.max(np.abs(recorded - immediate)))
    if not np.isfinite(error) or error <= 1e-6:
        raise ValueError("real EA mismatch was not reproduced")
    print(json.dumps(dict(
        status="recorded-order cache is not a causal shuffled-delivery input",
        person="P01", session=str(sessions[held[0]]), order="full0",
        first_global_row=int(delivered[0]), unavailable_recorded_prefix_rows=first,
        aligned_input_max_abs_difference=error, immediate_prefix_exact=True,
        held_labels_opened=False, accuracy_scored=False, input_sha256=PINS,
    ), indent=2))


if __name__ == "__main__":
    main()
