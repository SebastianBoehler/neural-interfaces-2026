"""Lane E helpers: load an exact package ZIP as the official Solver and stream Scherer windows through fresh copies.

OFFICIAL = /official on the cluster images, 2026-competition in the repo. Windows stream one per predict call (official
fresh_model per session); the model's [v6] stderr line is the per-session log (refits, over, ft_s, errors).
"""
import importlib.util
import sys
import time
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path("/official") if Path("/official").is_dir() else ROOT / "2026-competition"
sys.path.insert(0, str(OFFICIAL / "tracks/bci_decoding"))  # as niv/worker.py
from benchmark_utils.streaming import fresh_model  # noqa: E402


def load(zip_path, channels, device, tmp):
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp)
    spec = importlib.util.spec_from_file_location(f"pkg_{Path(zip_path).stem}", Path(tmp) / "submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Solver().load_model({"submission_dir": Path(tmp), "device": device, "ch_names": channels})


def scherer(path):
    with np.load(path, allow_pickle=False) as z:
        return torch.from_numpy(z["X"]), z["subject"], z["session"], z["ch_names"].tolist()


def stream_rows(subject, session, n, offset=0):
    """First n rows of the visits concatenated in sorted visit order (one 'session' of n windows; recording order)."""
    key = np.char.add(np.char.add(subject, "|"), session)
    rows = np.concatenate([np.flatnonzero(key == k) for k in sorted(set(key))])
    return rows[offset:offset + n]


def run(model, X, rows, device, sync=True):
    """Fresh copy, one window per call. -> labels, trace outs, seconds (copy + reset + every predict)."""
    t0 = time.perf_counter()
    m = fresh_model(model)
    labels, outs = [], []
    for r in rows:
        y = m.predict(X[int(r):int(r) + 1].to(device))
        labels.append(int(y[0]))
        outs.append(np.asarray(m.trace["out"], np.float64).copy())
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    return np.array(labels), np.stack(outs), time.perf_counter() - t0, m
