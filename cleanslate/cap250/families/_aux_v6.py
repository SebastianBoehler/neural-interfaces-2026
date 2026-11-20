"""Shared helpers for the eye and rest families (recipe of scripts/train_v6_members.py on source rows only)."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source import ROOT, family_args, load_source  # noqa: E402

os.environ.setdefault("NI26_WS", str(ROOT))
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "scripts")]
SMOKE = os.environ.get("CAP250_SMOKE") == "1"


def load_module(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("this family requires Slurm CUDA")


def write_fit(out, record):
    (out / "diagnostics").mkdir()
    (out / "diagnostics" / "fit.json").write_text(json.dumps(record, indent=2) + "\n")


def start():
    return time.time()
