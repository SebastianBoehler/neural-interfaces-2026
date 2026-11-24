"""Lane G gS on Lane A's fold packages: python gs_fold.py OUT_DIR  (run in the worker SIF on Galvani).

For P04 and P10: g4 fit (laneG/fit.py, same seed/replicas as the production gS) on the person's VISIT 1 ONLY, take own_s_W/c, and write
OUT/<who>_s23/ = the fold package with exactly that person's own_s row replaced (weights.npz new; every other file symlinked) and
OUT/<who>_ctl/ = same, own_s row rewritten with its own stock values (null control: must score exactly 0).
No visit 2/3 window is read (the production gS used all sessions, which would leak into the held visits).
"""
import os
import sys
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import common as K
import fit as F

BUNDLE = Path("/mnt/lustre/work/gehler/gfh527/ni26_js321_20261117_returning_bundle2")


def link_dir(src, dst, weights):
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name == "weights.npz":
            continue
        t = dst / f.name
        if not t.exists():
            if f.name == "submission.py":
                t.write_bytes(f.read_bytes())
            else:
                os.symlink(f.resolve(), t)
    np.savez(dst / "weights.npz", **weights)


def main(out):
    out = Path(out)
    D = K.public()
    for who in ("P04", "P10"):
        src = BUNDLE / f"{who}_s23" / "base"
        with TemporaryDirectory(dir="/work") as tmp:
            zp = Path(tmp) / "p.zip"
            with zipfile.ZipFile(zp, "w") as zf:
                for n in ("weights.npz", "submission.py"):
                    zf.write(src / n, n)
            mod, z = K.package(zp, Path(tmp) / "w")
        j = [str(n) for n in z["route_names"]].index(who)
        g4 = F.fit(mod, z, K.OwnFeat(z), D, who, (1,), "g4")  # visit 1 only
        new = dict(z)
        for k in ("own_s_W", "own_s_c"):
            assert g4[k].shape == z[k].shape[1:] and np.isfinite(g4[k]).all()
            new[k] = z[k].copy()
            new[k][j] = g4[k].astype(z[k].dtype)
        changed = [k for k in z if not np.array_equal(z[k], new[k])]
        assert sorted(changed) == ["own_s_W", "own_s_c"], changed
        d = {k: float(np.abs(new[k][j] - z[k][j]).max()) for k in changed}
        print(who, "row", j, "max abs change of own_s row", d, flush=True)
        link_dir(src, out / f"{who}_s23", new)
        link_dir(src, out / f"{who}_ctl", dict(z))


if __name__ == "__main__":
    main(sys.argv[1])
