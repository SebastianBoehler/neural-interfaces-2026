"""Data-only variant of a fold package: hardlink every file, rewrite weights.npz (and donor files) only.

python scripts/redteam/patch_weights.py SRC_PKG OUT_PKG [--donor PKG --take key1,key2,file.pt,...] [--set key=value ...]
--take copies arrays (names in weights.npz) and whole files (names with a dot) from the donor package;
--set overwrites a scalar array. Every named key must exist in SRC with the same shape and dtype, every value must
change, and nothing else may change: otherwise the script stops (no silent no-op). submission.py is never touched."""
import argparse
import os
import shutil
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("src", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--donor", type=Path)
ap.add_argument("--take", default="")
ap.add_argument("--set", nargs="*", default=[])
a = ap.parse_args()
names = [n for n in a.take.split(",") if n]
files, keys = [n for n in names if "." in n], [n for n in names if "." not in n]
assert not names or a.donor, "--take needs --donor"
a.out.mkdir(parents=True)
for f in a.src.iterdir():
    if f.name != "weights.npz" and f.name not in files:
        os.link(f, a.out / f.name)
for name in files:
    assert (a.src / name).is_file() and (a.donor / name).read_bytes() != (a.src / name).read_bytes(), f"{name}: missing or already equal"
    shutil.copyfile(a.donor / name, a.out / name)
with np.load(a.src / "weights.npz", allow_pickle=False) as z:
    w = {k: z[k] for k in z.files}
before = dict(w)
if keys:
    with np.load(a.donor / "weights.npz", allow_pickle=False) as d:
        for k in keys:
            assert k in w and d[k].shape == w[k].shape and d[k].dtype == w[k].dtype, f"{k}: shape or dtype differs"
            w[k] = d[k].copy()
for item in a.set:
    k, v = item.split("=")
    assert k in w and w[k].shape == (), f"{k}: not a scalar key"
    w[k] = np.asarray(float(v), dtype=w[k].dtype)
changed = sorted(k for k in w if not np.array_equal(w[k], before[k]))
assert changed == sorted(keys + [i.split("=")[0] for i in a.set]), f"changed {changed}"
np.savez(a.out / "weights.npz", **w)
print("built", a.out, "arrays changed:", changed, "files replaced:", files)
