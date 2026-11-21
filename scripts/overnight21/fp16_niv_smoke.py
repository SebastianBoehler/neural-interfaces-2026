"""Verify the pinned NI-V replay before clearing the smoke paths for a fresh order."""
from pathlib import Path
import sys

import numpy as np


def check(stage, reference, current):
    directories = {
        "smoke-new": ("lead", "trial"),
        "smoke-ret": ("ret_lead_k5", "ret_trial_k5"),
    }[stage]
    for directory in directories:
        name = Path(directory) / "smoke_5.npz"
        old_path, new_path = reference / name, current / name
        with np.load(old_path, allow_pickle=False) as old, np.load(new_path, allow_pickle=False) as new:
            expected_streams = 2 if stage == "smoke-new" else 1
            if old.files != new.files or len(new.files) != 3 * expected_streams:
                raise ValueError(f"Pinned smoke stream set changed: {name}")
            for key in old.files:
                limit = slice(3) if key.endswith("|meta") else slice(None)
                np.testing.assert_array_equal(old[key][limit], new[key][limit], err_msg=f"{name}: {key}")
        new_path.rename(new_path.with_name("pinned_71840.npz"))
    print(f"OK: {stage} pinned 71840 rows, labels and activity")


if __name__ == "__main__":
    check(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]))
