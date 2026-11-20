"""Authenticate the source row scope and file inventory of every Slurm family job."""

import json
import os

from components import expected_files
from source import file_digest


def files(path, family, seed, base_only):
    names = expected_files(family, seed)
    if base_only and family == "geometry":
        names = ("weights.npz",)
    return {name: file_digest(path / name) for name in names}


def read_receipt(path, family, seed, row_hash, data_hash, *, base_only=False):
    with (path / "receipt.json").open() as handle:
        receipt = json.load(handle)
    expected = dict(family=family, seed=seed, row_sha256=row_hash, data_sha256=data_hash,
                    base_only=base_only, files=files(path, family, seed, base_only))
    if receipt.get("smoke") != (os.environ.get("CAP250_SMOKE") == "1"):
        raise ValueError(f"reduced/full fit mismatch: {path}")
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError(f"fit receipt mismatch: {path}")
    return receipt
