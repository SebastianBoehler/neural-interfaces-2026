"""Pratibha-2 load and forward smoke on a compute node: strict load, positions in cm, shapes, bf16 vs fp32 agreement."""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "laneC")]
import pratibha  # noqa: E402
from oracle import require  # noqa: E402

root, checkpoint = Path(sys.argv[1]), Path(sys.argv[2])
member = pratibha.Member(checkpoint).cuda().eval()
n = sum(p.numel() for p in member.body.parameters())
print("body parameters", n)
require(abs(n - 92_814_848) < 200_000, f"parameter count {n} differs from the 92.8M release")  # buffer freq_matrix excluded
with np.load(root / "data/proteus_windows.npz", allow_pickle=False) as z:
    names, x = z["ch_names"].tolist(), torch.from_numpy(z["X"][:8].astype(np.float32)).cuda()
pos = member.positions(names, "cuda")
print("Cz cm", member.table["Cz"], "->", pos[0, names.index("Cz")].tolist() if "Cz" in names else "n/a", "radius range",
      pos.norm(dim=-1).min().item(), pos.norm(dim=-1).max().item())
require(5 < pos.norm(dim=-1).min().item() and pos.norm(dim=-1).max().item() < 14, "positions are not in cm")
with torch.inference_mode():
    f32, m32 = member(x, pos)
    feats = member.body(torch.randn(2, 41, 800, device="cuda"), pos.expand(2, -1, -1))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        f16, _ = member(x, pos)
print("features", tuple(feats.shape), "feature std over tokens", feats.float().std(dim=(1, 2)).mean().item())
require(tuple(feats.shape) == (2, 4, 41, 512) and bool(torch.isfinite(feats).all()), "bad feature tensor")
print("task logits fp32", f32[0].tolist(), "max |bf16 - fp32|", (f16.float() - f32).abs().max().item())
require(torch.isfinite(f32).all() and (f16.float() - f32).abs().max() < 0.5, "bf16 forward disagrees with fp32")
print("PASS smoke")
