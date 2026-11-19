"""Assemble the worker's single source file and authenticate local initialization."""
import importlib.util
import json
import shutil
from pathlib import Path

import torch

from data import sha

HERE = Path(__file__).resolve().parent
HUB = Path.home() / ".cache/huggingface/hub"
BASE = {
    "reve": (
        "models--brain-bzh--reve-base/snapshots/dc2a075c287bb2f6c04ee5875bd79535a0f7dba6",
        "8ecc650619598748286c2457f81f5c6bd12e8bb59db44f7b02af1955c44de8fe",
    ),
    "cbramod": (
        "models--braindecode--cbramod-pretrained/snapshots/584cdc415913739a05d84bf0c1cb3db397764507",
        "a939ace9aa1e229f08391ad8bb2d197b507ae2c519a50addf087f0151b2df5c3",
    ),
    "manas": ("", "5c7fb0a1de75bf01c179fdc416a4e4af3ad4ead7895ac03b7e357adb587cfa64"),
}
POSITION_SHA = "e7fb3cbd81a7f20b8e2714a4c01681d250fb17b8984e1443735ae8e3ced062d5"


def bundle(arch, out, assets, recipe):
    out.mkdir(parents=True, exist_ok=False)
    files = ["eeg", "backbone", "online", "solver"]
    if arch == "manas":
        files.insert(1, "manas")
        shutil.copyfile(assets / "LICENSE", out / "LICENSE-MANAS.txt")
    if recipe == "A2":
        files.insert(-2, "spectral")
    target = out / "submission.py"
    target.write_text("\n\n".join((HERE / f"{name}.py").read_text() for name in files))
    return load_source(target)


def load_source(path):
    spec = importlib.util.spec_from_file_location("team_a_submission", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def initialize(runtime, arch, names, assets, recipe, checkpoint=None):
    from safetensors.torch import load_file

    subdir, expected = BASE[arch]
    path = checkpoint or ((assets if arch == "manas" else HUB / subdir) / "model.safetensors")
    if sha(path) != expected:
        raise ValueError(f"{arch}: incorrect pretrained checkpoint digest")
    positions = None
    if arch == "manas":
        pos = assets / "positions.json"
        if sha(pos) != POSITION_SHA:
            raise ValueError("Incorrect MANAS-2 electrode table digest")
        table = json.loads(pos.read_text())
        positions = torch.tensor([table[n] for n in names], dtype=torch.float32) * 100
    decoder = runtime.SpectralDecoder if recipe == "A2" else runtime.Decoder
    model = decoder(arch, names, positions)
    weights = load_file(path)
    if arch == "reve":
        ignored = {"cls_query_token"} | {k for k in weights if k.startswith("final_layer.")}
    elif arch == "manas":
        prefixes = ("decoder.", "band_decoder.", "band_target_module.", "band_pooler.", "aux_linear.", "aux_predict.")
        ignored = {"aux_query"} | {k for k in weights if k.startswith(prefixes)}
    else:
        ignored = set()
    model.encoder.load_state_dict({k: v for k, v in weights.items() if k not in ignored}, strict=True)
    return model, expected
