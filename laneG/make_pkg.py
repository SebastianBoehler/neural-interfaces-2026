"""Splice Lane G personal arrays into a byte-for-byte copy of the sealed cap250 ZIP.

python make_pkg.py CAP250.zip PROD_DIR OUT.zip [MODE=g1]
Replaces only route_ps1_W/c and the own_* rows (people in route_names order) by PROD_DIR/<person>_<mode>.npz; every other array,
every other archive member (submission.py included) is copied unchanged and asserted equal. Gate and gallery arrays are
untouched, so recognition is identical to stock by construction. Also times the personal stage with stock vs new rows.
"""
import hashlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

import common as K

KEYS = ("route_ps1_W", "route_ps1_c", "own_prior", "own_csp", "own_t_mu", "own_t_sd", "own_t_W", "own_t_c", "own_c_W", "own_c_c",
        "own_s_W", "own_s_c")


def main(zip_in, prod, zip_out, mode="g1"):
    mod, z = K.package(zip_in, Path(zip_out).with_suffix(".work"))
    names = [str(n) for n in z["route_names"]]
    new = dict(z)
    keys = ("own_s_W", "own_s_c") if mode == "gS" else KEYS  # gS: only the slow-course member, from the g4 fit
    for j, name in enumerate(names):
        row = np.load(Path(prod) / f"{name}_{'g4' if mode == 'gS' else mode}.npz")
        for k in keys:
            assert row[k].shape == z[k].shape[1:], (name, k, row[k].shape, z[k].shape)
            assert np.isfinite(row[k]).all(), (name, k, "not finite")
            new[k] = new[k].copy() if j == 0 else new[k]
            new[k][j] = row[k].astype(z[k].dtype)
    changed = [k for k in z if not np.array_equal(z[k], new[k])]
    assert set(changed) <= set(keys), f"unexpected changes {set(changed) - set(keys)}"
    assert all(k in changed for k in keys if k != "own_prior"), "a personal array did not change: silent no-op"
    for k in ("route_gal", "route_names", "bio_gal", "bio_m", "v6_tpri", "own_bands", "own_beta", "own_kref", "route_slot_g"):
        assert np.array_equal(z[k], new[k]), f"{k} (recognition / dose) changed"
    mod.check_package(new)
    buf = io.BytesIO()
    np.savez(buf, **new)
    with zipfile.ZipFile(zip_in) as zi, zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zo:
        for info in zi.infolist():
            data = buf.getvalue() if info.filename == "weights.npz" else zi.read(info.filename)
            zo.writestr(info, data)
    with zipfile.ZipFile(zip_in) as a, zipfile.ZipFile(zip_out) as b:
        assert [i.filename for i in a.infolist()] == [i.filename for i in b.infolist()]
        same = [i.filename for i in a.infolist() if i.filename != "weights.npz" and a.read(i) == b.read(i.filename)]
        assert len(same) == len(a.infolist()) - 1, "a non-weights member changed"
    D = K.public()
    X = D["X"][np.flatnonzero((D["person"] == "P04") & (D["visit"] == 2))[:80]].astype(np.float32)
    ms = {}
    for tag, zz in (("stock", z), ("new", new)):
        st = mod.OwnStage(zz)
        t0 = time.perf_counter()
        for x in X:
            st.observe(x)
            st.log_members(3)
        ms[tag] = 1e3 * (time.perf_counter() - t0) / len(X)
    receipt = dict(source_zip_sha256=hashlib.sha256(Path(zip_in).read_bytes()).hexdigest(),
                   out_zip_sha256=hashlib.sha256(Path(zip_out).read_bytes()).hexdigest(), mode=mode, replaced=sorted(changed),
                   submission_py_unchanged=True, personal_stage_ms_per_window=ms)
    Path(zip_out).with_suffix(".json").write_text(json.dumps(receipt, indent=1))
    print(json.dumps(receipt, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:5])
