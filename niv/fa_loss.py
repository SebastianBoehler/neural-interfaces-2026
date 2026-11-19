"""Accuracy cost of false recognition: stream of person j through package k (k != j, order full:0, both visits) minus the same
stream through the un-enrolled package (out/TAG).   python fa_loss.py OUT_DIR TAG DATA.npz"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bank  # noqa: E402

out, tag, data = sys.argv[1:4]
base = bank.assemble(Path(out) / tag, data)["cell"][0, 0]
d, n, rec = [], 0, 0
for k in range(9):
    r = bank.assemble(Path(out) / f"fa_{tag}_k{k}", data)
    m = np.isfinite(r["cell"][0, 0])
    m[k] = False
    d.append((r["cell"][0, 0] - base)[m])
    rec += int(((r["rec"][0, 0] >= 0) | (r["routed"][0, 0] > 0))[m].sum()); n += int(m.sum())
d = np.concatenate(d)
print(f"{tag}: recognised {rec}/{n} streams; accuracy fa - unenrolled: mean {d.mean():+.2f} pp, "
      f"min {d.min():+.2f}, max {d.max():+.2f}, share worse {np.mean(d < 0):.2f}")
