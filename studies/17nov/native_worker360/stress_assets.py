"""Prepare local-only stress artifacts with both recognition thresholds raised."""
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

KEYS = ("bio_tau", "route_tau")


def prepare(source, dest, filename):
    shutil.copytree(source, dest, copy_function=os.link,
                    ignore=lambda _path, names: set(names) & {filename, "__pycache__", "source_fixture.pt"})
    if filename == "native.pt":
        data = torch.load(source / filename, map_location="cpu", weights_only=True)
        for key in KEYS:
            data[key] = torch.full_like(data[key], 1e6)
        torch.save(data, dest / filename)
    else:
        with np.load(source / filename, allow_pickle=False) as file:
            data = dict(file)
        for key in KEYS:
            data[key] = np.full_like(data[key], 1e6)
        np.savez(dest / filename, **data)


if __name__ == "__main__":
    reference, candidate = map(Path, sys.argv[1:])
    prepare(reference, Path("stress_reference"), "weights.npz")
    prepare(candidate, Path("stress_native"), "native.pt")
