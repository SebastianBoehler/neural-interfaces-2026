"""Lane D3: add the enrolment moments (d3_mu, d3_var, d3_n) of ONE person of a fold package, whose visit 1 the package holds as its single session.
usage: add_d3.py PKG_DIR OUT_DIR WHO DATA.npz     (WHO = P04 | P10; runs in the worker image)
OUT_DIR = copy of PKG_DIR (symlinks) with a new weights.npz: other people NaN rows. Self-check: the person's contrast-gallery row recomputed from the same
views must correlate > 0.97 with the stored route_gal row (smoke 20 Nov: relative max error 0.17-0.32, correlation 0.99; the fold's stored row was not built with this package's
own tangent parameters, so no exact match; the moments are in the package's own query-side coordinate system, which is the one the drift stage needs)."""
import ast
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "niv"))
import enrol  # noqa: E402


def main(pkg, out, who, data):
    pkg, out = Path(pkg), Path(out)
    z = dict(np.load(pkg / "weights.npz", allow_pickle=False))
    with np.load(data, allow_pickle=False) as d:
        sel = np.flatnonzero((d["subject"] == f"Dreyer2026Proteus/{who}") & (d["session"] == "1"))
        sel = sel[np.lexsort((d["onset"][sel], d["run"][sel]))]
        X, y = d["X"][sel], d["y"][sel].astype(int)
    names = [str(n) for n in z["route_names"]]
    hit = [i for i, n in enumerate(names) if n.endswith(who)]
    assert len(hit) == 1, (who, names)
    i = hit[0]
    U = enrol.views(pkg, z, X)
    ns = {"np": np, "CLIP": 4.0}
    enrol._functions(ROOT / "replay/rlib2.py", ("rstats", "rz", "contrast", "onehot"), ns)
    gal = ns["contrast"](U, ns["onehot"](y), 256)
    err = float(np.abs(gal - z["route_gal"][i]).max() / max(np.abs(z["route_gal"][i]).max(), 1e-9))
    print(who, "index", i, "windows", len(y), "gallery relative max error", err, flush=True)
    corr = float(np.corrcoef(gal.ravel(), z["route_gal"][i].ravel())[0, 1])
    print(who, "gallery correlation", corr, flush=True)
    assert corr > 0.97, "the recomputed views do not resemble the stored enrolment row"
    rows = enrol.d3_rows(U, y)
    n0 = len(names)
    for k, shape in zip(enrol.D3, ((3, 12), (3, 12), (3,))):
        z[k] = np.full((n0, *shape), np.nan)
        z[k][i] = rows[k][0]
    out.mkdir(parents=True)
    for f in pkg.iterdir():
        if f.name != "weights.npz":
            (out / f.name).symlink_to(f.resolve())
    np.savez(out / "weights.npz", **z)
    print("wrote", out, "d3_n", rows["d3_n"][0], flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
