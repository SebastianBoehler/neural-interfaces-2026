"""Pinned inputs and artifact checks for the independent F5 evaluator."""

import hashlib
import json
from pathlib import Path


OBJECTIVES = ("true", "wrongswap", "ordinary")
SEEDS = (460, 1460)
MODELS = tuple(f"{kind}_{seed}" for kind in OBJECTIVES for seed in SEEDS)
ARMS = ("stock",) + tuple(f"{name}_{mode}" for name in MODELS for mode in ("genuine", "blank"))
ORDERS = (0, 1)
PUBLIC_DATA = "2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327"
PUBLIC_ORDERS = "6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f"
SOURCE_PEOPLE = [f"P{i:02}" for i in range(1, 11)]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_models(directory):
    directory = Path(directory)
    binding = json.loads((directory / "model_binding.json").read_text())
    assert tuple(sorted(binding)) == ("actor", "teacher")
    assert all(len(x) == 64 and len(bytes.fromhex(x)) == 32 for x in binding.values())
    manifest = directory / "models.sha256"
    files = {name: value for value, name in (row.split() for row in manifest.read_text().splitlines())}
    assert set(files) == {f"{name}.{suffix}" for name in MODELS for suffix in ("pt", "json")}
    for name, expected in files.items():
        assert digest(directory / name) == expected, name
    for name in MODELS:
        kind, seed = name.split("_")
        info = json.loads((directory / f"{name}.json").read_text())
        assert info["objective"] == kind and info["seed"] == int(seed), name
        assert info["checkpoint"] == files[f"{name}.pt"], name
        assert info["architecture"] == "chronological" and info["steps"] == 600, name
        assert info["data"] == PUBLIC_DATA and info["orders"] == PUBLIC_ORDERS, name
        assert info["train_people"] == SOURCE_PEOPLE and info["external_labels_opened"] is False, name
        assert all(info[k] == binding[k] for k in binding), name
    return files
