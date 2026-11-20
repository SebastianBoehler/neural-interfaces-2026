"""Compare fresh all-ten eye/rest/cbramod fits with the archived artifacts.

usage: _aux_compare.py NEW_OUT_DIR ARCHIVE_V6_DIR ARCHIVE_CB_CKPT_DIR   (CPU job; NEW_OUT_DIR has family/sN/ parts)"""

import hashlib
import io
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "submissions/final_stack"))
from cb_export import export_blob  # noqa: E402


def sha(b):
    return hashlib.sha256(b).hexdigest()[:12]


def tensors(path):
    m = torch.jit.load(str(path), map_location="cpu")
    return {k: v.double() for k, v in m.state_dict().items()}


def diff(a, b):
    if a.keys() != b.keys():
        return float("inf")
    return max(float((a[k] - b[k]).abs().max()) if a[k].numel() else 0.0 for k in a)


def main(new, v6, cbs):
    new, v6, cbs = Path(new), Path(v6), Path(cbs)
    for s in range(3):
        n, a = new / f"eye/s{s}/eye_s{s}.pt", v6 / f"eye_s{s}.pt"
        print(f"eye s{s} bitwise={n.read_bytes() == a.read_bytes()} sha {sha(n.read_bytes())}/{sha(a.read_bytes())} max|dw|={diff(tensors(n), tensors(a)):.3g}")
    npz = []
    for s in range(3):
        n, a = new / f"rest/s{s}/rest_s{s}.pt", v6 / f"rest_s{s}.pt"
        print(f"rest s{s} bitwise={n.read_bytes() == a.read_bytes()} max|dw|={diff(tensors(n), tensors(a)):.3g}")
        z, y = np.load(new / f"rest/s{s}/rest_s{s}.npz"), np.load(v6 / f"rest_s{s}.npz")
        print("  npz max|d|", {k: float(np.abs(z[k].astype(float) - y[k].astype(float)).max()) for k in y.files})
        npz.append(z)
    print("rest arrays equal across new seeds:", all(np.array_equal(npz[0][k], z[k]) for z in npz[1:] for k in npz[0].files))
    for s in range(3):
        ckpt = torch.load(cbs / f"reve_full_C_{s}_all.pt", map_location="cpu", weights_only=True)
        ref, got = export_blob(ckpt, half=True), torch.load(new / f"cbramod/s{s}/cb_s{s}.pt", map_location="cpu", weights_only=True)
        worst = max(float((got["net"][k].double() - ref["net"][k].double()).abs().max()) for k in ref["net"])
        head = max(float((got[k].double() - ref[k].double()).abs().max()) for k in ("head_w", "head_b", "prior"))
        buf = io.BytesIO()
        torch.save(ref, buf)
        print(f"cb s{s} tag {got['tag']}/{ref['tag']} net max|dw|={worst:.3g} head/prior max|d|={head:.3g}")


if __name__ == "__main__":
    main(*sys.argv[1:])
