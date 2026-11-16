"""Run v6.6's personalisation stage on a §281 arm's own delivered posteriors.

Usage: python -I studies/14nov/newmember/s281_cp.py ARM ORDER BASE
The corrected chain output is s281_v66_<ARM>_<ORDER>.npz under runs/chainout.
This produces per-window scores, not a metric or a candidate package.
"""
import hashlib
import importlib.util
import io
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("cpers", ROOT / "lean/src66/cpers.py")
CP = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CP)
CONFIG = {"v6_cp_fs": 120.0, "v6_cp_t0": 20, "v6_cp_k": 20, "v6_cp_w0": 90,
          "v6_cp_R": 120, "v6_cp_cap": 0.10, "v6_cp_q": 0.5,
          "v6_cp_bands": np.array([(4, 8), (8, 12), (12, 16), (16, 20),
                                    (20, 24), (24, 28), (28, 32), (32, 40)], float)}


def apply_sessions(windows, raw, filtered, sessions, config=CONFIG):
    """Return scores in input-row order; start a fresh package stage per session."""
    n = len(windows)
    assert raw.shape == filtered.shape == (n, 3)
    assert np.isfinite(raw).all() and np.isfinite(filtered).all()
    assert np.allclose(raw.sum(1), 1) and np.allclose(filtered.sum(1), 1)
    rows = np.concatenate(sessions)
    assert len(rows) == n and np.array_equal(np.sort(rows), np.arange(n)), "sessions must partition rows exactly"
    result = np.empty((n, 3), dtype=np.float64)
    for session in sessions:
        cp = CP.CausalPers(config)
        for row in session:
            result[row] = cp.step(windows[row], raw[row], filtered[row], False)
        assert cp.log["err"] == 0, f"personalisation failed: {cp.log}"
        assert cp.stopped is None and cp.t == len(session)
    assert np.isfinite(result).all()
    return result


def verify_package_config():
    package = ROOT / "runs/sealed/final_zips/v66_cp1_cap250.zip"
    with zipfile.ZipFile(package) as z, np.load(io.BytesIO(z.read("weights.npz"))) as weights:
        for key, value in CONFIG.items():
            assert np.array_equal(weights[key], value), f"package configuration differs: {key}"
        assert int(weights["v6_cp_on"]) == 1 and int(weights["v6_ft_cap"]) == 250


def main():
    arm, order, base = sys.argv[1], sys.argv[2], sys.argv[3]
    assert arm in {"control", "S1", "S2", "S3", "S4", "S5"}
    verify_package_config()
    sys.path.insert(0, str(ROOT / "studies/14nov/rundetect"))
    import package as K

    assert np.array_equal(K.ROWS, np.arange(K.C.N)), "unexpected replay row layout"
    src = ROOT / f"runs/chainout/s281_v66_{arm}_{order}.npz"
    audit = ROOT / f"runs/newmember/s281_v66_{arm}_{order}_gate.npz"
    assert audit.is_file(), f"missing gate verification: {audit}"
    dst = ROOT / f"runs/s281_v66_cp/{arm}_{order}_{base}.npz"
    assert not dst.exists(), f"refusing to overwrite {dst}"
    with np.load(src) as z:
        raw = z[f"{base}|V6Fr"].astype(np.float64)
    with np.load(audit) as z:
        assert np.array_equal(raw, z[f"{base}|V6Fr"]), "chain output differs from gate-verified output"
    assert raw.shape == (K.C.N, 3)
    lat = K.latch(order)
    hc = K.J.hyp_cache(order, "wprior", K.X, K.LAM, sessions=K.SESS, proto=K.PROTO)
    filtered = K.gate(raw, K.J.filt_joint(raw, hc, "urn"), order, lat)
    sessions = [K.C.order_rows(order, s, K.PROTO) for s in K.SESS]
    with np.load(ROOT / "data/proteus_windows.npz") as data:
        windows = data["X"].astype(np.float64)
    result = apply_sessions(windows, raw, filtered, sessions)
    dst.parent.mkdir(parents=True, exist_ok=True)
    stage = ROOT / "lean/src66/cpers.py"
    np.savez_compressed(dst, scores=result, source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                        stage_sha256=hashlib.sha256(stage.read_bytes()).hexdigest(), arm=arm, order=order, base=base)
    print(f"{dst}: {len(result)} windows, {len(sessions)} independent sessions; package stage errors 0")


if __name__ == "__main__":
    main()
