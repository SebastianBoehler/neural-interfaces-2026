"""Swap one ShallowConvNet member of a package for a TorchScript twin.

Usage: python scripts/export_jit_member.py SRC_DIR OUT_DIR [--member I]
Proves the generic ``Jit`` member of submissions/ensemble/submission.py end to
end (31 Oct, before the code freeze): a scripted module re-implementing
``submission.Shallow`` is saved as ``memberI.pt`` and referenced by
``netI_jit_file``; the member's alignment keys stay, so it keeps its input
stream. Checks, on all 2,655 public windows: the twin's probabilities equal
Shallow's (float32 tolerance), and the package streams to the source labels.
Uses the current submission.py; writes OUT_DIR_check.json.
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "2026-competition/tracks/bci_decoding"))

import protocol  # noqa: E402
from add_rebalance import SRC, load_solver, stream  # noqa: E402

ALIGN = ("align", "k", "G", "scale", "shrink")


class ShallowTS(nn.Module):
    """submission.Shallow as a module: (B, C, T) float32 -> logits (B, 3)."""

    def __init__(self, p):
        super().__init__()
        t = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in p.items() if k not in ALIGN}
        self.register_buffer("w", (t["time_w"] * t["spat_w"].permute(1, 0, 2, 3)).sum(0).unsqueeze(1))
        self.register_buffer("b", t["spat_w"].squeeze().sum(-1).mm(t["time_b"].unsqueeze(-1)).squeeze())
        for k in ("bn_mean", "bn_var", "bn_w", "bn_b", "cls_w", "cls_b"):
            self.register_buffer(k, t[k])

    def forward(self, x):
        h = F.conv2d(x.unsqueeze(1).transpose(2, 3), self.w, self.b)
        h = F.batch_norm(h, self.bn_mean, self.bn_var, self.bn_w, self.bn_b, False, 0.0, 1e-5)
        h = F.avg_pool2d(h * h, (75, 1), (15, 1))
        h = torch.log(torch.clamp(h, min=1e-6))
        return F.conv2d(h, self.cls_w, self.cls_b)[:, :, 0, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--member", type=int, default=0)
    a = ap.parse_args()
    with np.load(a.src / "weights.npz") as f:
        z = {k: f[k] for k in f.files}
    pre = f"net{a.member}_"
    p = {k[len(pre):]: v for k, v in z.items() if k.startswith(pre)}
    assert "time_w" in p, "member is not a ShallowConvNet"
    a.out.mkdir(parents=True, exist_ok=True)
    name = f"member{a.member}.pt"
    torch.jit.save(torch.jit.script(ShallowTS(p).eval()), str(a.out / name))
    zj = {k: v for k, v in z.items() if not (k.startswith(pre) and k[len(pre):] not in ALIGN)}
    zj[pre + "jit_file"] = np.array(name)
    np.savez(a.out / "weights.npz", **zj)
    shutil.copyfile(SRC, a.out / "submission.py")

    d = protocol.load()
    old, new = load_solver(a.src, "src_pkg"), load_solver(a.out, "jit_pkg")
    X = d["X"].astype(np.float32)
    ref = next(n for _, ns in old.groups.values() for i, n in ns if i == a.member)
    twin = next(n for _, ns in new.groups.values() for i, n in ns if i == a.member)
    assert type(twin).__name__ == "Jit", type(twin).__name__
    dp = max(float(np.abs(ref.proba(X[i:i + 256]) - twin.proba(X[i:i + 256])).max())
             for i in range(0, len(X), 256))
    rows = np.ones(len(X), bool)
    lo = stream(old, d, rows, lambda m, x: m.predict(x))
    ln = stream(new, d, rows, lambda m, x: m.predict(x))
    sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
    res = {"source": str(a.src), "member": a.member, "max_abs_proba_diff": dp,
           "labels_equal": float(np.mean(lo == ln)), "n_windows": int(len(lo)),
           "files_sha256": {f.name: sha(f) for f in sorted(a.out.iterdir()) if f.is_file()}}
    (a.out.parent / f"{a.out.name}_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    assert dp < 1e-5 and res["labels_equal"] == 1.0


if __name__ == "__main__":
    main()
