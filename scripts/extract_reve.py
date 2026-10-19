"""Extract frozen REVE embeddings for every exported PROTEUS window.

Mirrors the official ``solvers/reve_probe.py`` encoder: linear resampling to
200 Hz, per-window per-channel z-scoring, clamping to +-15, REVE features
averaged over channels, then mean-pooled over time (the probe's pooling).
Writes data/feature_cache/reve_base.npy with shape (N, D). Each window is
encoded alone, so the features carry no cross-window information.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from braindecode.models import REVE

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import protocol  # noqa: E402

OUT = ROOT / "data/feature_cache/reve_base.npy"


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    d = protocol.load()
    names = [str(n) for n in d["ch_names"]]
    model = REVE.from_pretrained("brain-bzh/reve-base", n_outputs=1, n_chans=41,
                                 n_times=800, sfreq=200, chs_info=None)
    model.default_pos = model.get_positions(names)
    model = model.to(dev).eval().requires_grad_(False)
    feats, t0 = [], time.time()
    for i in range(0, len(d["X"]), 32):
        X = torch.as_tensor(d["X"][i:i + 32]).to(dev)
        X = F.interpolate(X, size=800, mode="linear", align_corners=False)
        X = (X - X.mean(-1, keepdim=True)) / X.std(-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        with torch.inference_mode():
            out = model(X.clamp(-15.0, 15.0), return_features=True)
        f = out["features"] if isinstance(out, dict) else out
        if f.ndim == 4:
            f = f.mean(1)
        feats.append(f.mean(1).float().cpu().numpy())
        if i % 640 == 0:
            print(i, round(time.time() - t0, 1), "s", flush=True)
    F_ = np.concatenate(feats)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT, F_)
    print("saved", F_.shape, round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
