"""Lane FT-B6s stage C: stock fold packages with the six Sinc traces and their package arrays replaced by a re-fitted family.
usage: ftb_pack.py STOCK_PKG_RUN PARTS_DIR DEST_RUN ARM   STOCK_PKG_RUN holds stock/<fold>/pkg; PARTS_DIR holds <fold>/s<seed>/{trace,weights.npz}.
Hardlinks everything except the six Sinc traces and weights.npz. Fails loudly unless every replaced key exists with the same dtype and shape,
no key outside net53..58_{align,k,G,scale,shrink,sjit_file} changes, and the trace names are unchanged."""
import os
import sys
from pathlib import Path

import numpy as np

src, parts, dst, arm = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
PREFIX = "sbn_sincshallow_ea_ptm7r_rot05_k20"
folds = sorted(d.name for d in (src / "stock").iterdir() if (d / "pkg").is_dir())
assert len(folds) == 12, folds
SUFFIX = ("align", "k", "G", "scale", "shrink", "sjit_file")
for f in folds:
    out = dst / arm / f / "pkg"
    assert not out.exists(), out
    out.mkdir(parents=True)
    swapped = {f"{PREFIX}_s{s}.pt" for s in range(6)} | {"weights.npz"}
    for p in (src / "stock" / f / "pkg").iterdir():
        if p.name not in swapped:
            os.link(p, out / p.name)
    with np.load(src / "stock" / f / "pkg" / "weights.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    new = dict(arrays)
    for s in range(6):
        part = parts / f / f"s{s}"
        name = f"{PREFIX}_s{s}.pt"
        assert (part / name).is_file(), part / name
        os.link(part / name, out / name)
        with np.load(part / "weights.npz", allow_pickle=False) as z:
            for suf in SUFFIX:
                key = f"net{53 + s}_{suf}"
                assert key in arrays and key in z.files, key
                assert z[key].dtype == arrays[key].dtype and z[key].shape == arrays[key].shape, (key, z[key].dtype, arrays[key].dtype)
                new[key] = z[key]
            assert str(z[f"net{53 + s}_sjit_file"]) == name
    np.savez(out / "weights.npz", **new)
    with np.load(out / "weights.npz", allow_pickle=False) as z:
        changed = sorted(k for k in arrays if z[k].tobytes() != arrays[k].tobytes())
    assert changed and all(k.startswith(tuple(f"net{53 + s}_" for s in range(6))) for k in changed), changed
    assert set(os.listdir(out)) == set(os.listdir(src / "stock" / f / "pkg"))
    print(arm, f, "replaced", len(changed), "arrays", flush=True)
print("packed", arm)
