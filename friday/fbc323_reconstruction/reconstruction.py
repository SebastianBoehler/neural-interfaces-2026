"""Immutable sensor maps and equal, deterministic alternating source views."""
import hashlib
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
MAP = "b4128c37dacfef9ddb38d960b314f21ee98715b57c1120ddb60e98d9a80f4da4"
MEASURED = "P2 Pz CPz Cz P1 P3 P5 CP3 C3 P6 P4 CP4 C4 FC4 F4 F3 Fz FC3 FCz".split()
KINDS = ("hard", "alternating", "consistent")
MODES = ("ordinary", "bounded")
ARMS = tuple(f"{k}_{m}" for k in KINDS for m in MODES)
CONSISTENCY = .20


def geometry(channels):
    path = HERE / "maps.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != MAP:
        raise ValueError("the fixed physical reconstruction differs")
    with np.load(path, allow_pickle=False) as z:
        keep = z["keep"]
        if (not np.array_equal(z["channels"], channels) or z["retained"].tolist() != MEASURED or
                not np.array_equal(channels[keep], MEASURED)):
            raise ValueError("the exact 19 common recorded channels or official tensor order changed")
        maps = (z["spherical"], z["local"])
    for m in maps:
        if (m.shape != (41, 19) or m.dtype != np.float32 or not np.isfinite(m).all() or
                not np.array_equal(m[keep], np.eye(19)) or not np.allclose(m.sum(1), 1, atol=2e-6, rtol=0)):
            raise ValueError("one of the declared reconstruction maps is not constant-preserving")
    if np.array_equal(*maps):
        raise ValueError("the two reconstruction methods are identical")
    return keep, maps


@torch.no_grad()
def apply_maps(x, keep, maps):
    retained = x[:, keep]
    output = tuple(torch.as_tensor(m, device=x.device) @ retained for m in maps)
    for view in output:
        view[:, keep] = retained
        if view.shape != x.shape or not torch.equal(view[:, keep], retained) or not torch.isfinite(view).all():
            raise ValueError("actual source views did not keep the measured electrodes exactly")
    left, right = output
    gap = float((left - right).abs().max())
    if gap <= 1e-4:
        raise ValueError("the reconstructed sensors did not move on source EEG")
    return output, gap


def view_tape(batches, held, seed):
    rng = np.random.default_rng(21021 + 100 * int(held[1:]) + seed)
    tape, digest = [], hashlib.sha256()
    for b in (rows for epoch in batches for rows in epoch):
        if len(b) % 8:
            raise ValueError("the original session batch does not permit exact balanced dose")
        order = rng.permutation(len(b))
        n = len(b) // 8
        codes = np.full(len(b), -1, dtype=np.int8)
        codes[order[:n]] = 0
        codes[order[n:2*n]] = 1
        tape.append(codes)
        digest.update(codes.tobytes())
    return tape, digest.hexdigest()
