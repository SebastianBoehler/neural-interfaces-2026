"""Real-window parity for native common-reference filtering."""
import ctypes
import subprocess
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt_zi, sosfiltfilt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BANDS = [(1, 3), (2, 4), (3, 6), (4, 8), (6, 10), (8, 12), (10, 14), (12, 16),
         (14, 18), (16, 20), (18, 24), (20, 28), (24, 32), (28, 36), (32, 40),
         (36, 45), (40, 58)]


def test_real_window_car_bandpass_and_slow(tmp_path):
    source = HERE / "preprocess.cpp"
    assert source.is_file(), "native window preprocessor is missing"
    library = tmp_path / "preprocess.so"
    subprocess.run(["c++", "-std=c++17", "-O2", "-shared", "-fPIC", str(HERE / "filter.cpp"),
                    str(source), "-o", str(library)], check=True)
    lib = ctypes.CDLL(str(library))
    pf, pd = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_double)
    lib.nc_preprocess.argtypes = [pf, pd, pd, pd, pd, pf, pf]
    lib.nc_preprocess.restype = ctypes.c_int
    with np.load(ROOT / "data/proteus_windows.npz", allow_pickle=False) as z:
        x = np.ascontiguousarray(z["X"][0], dtype=np.float32)
    filters = np.ascontiguousarray([butter(4, list(b), btype="band", fs=120, output="sos") for b in BANDS])
    zi = np.ascontiguousarray([sosfilt_zi(s) for s in filters])
    slow = np.ascontiguousarray(butter(4, 1, btype="low", fs=120, output="sos"))
    slow_zi = np.ascontiguousarray(sosfilt_zi(slow))
    output = np.empty((17, 41, 480), np.float32)
    slow_output = np.empty((41, 8), np.float32)
    args = [x.ctypes.data_as(pf), filters.ctypes.data_as(pd), zi.ctypes.data_as(pd),
            slow.ctypes.data_as(pd), slow_zi.ctypes.data_as(pd), output.ctypes.data_as(pf),
            slow_output.ctypes.data_as(pf)]
    assert lib.nc_preprocess(*args) == 0
    ref = (x - x.mean(axis=0, keepdims=True)).astype(np.float32)
    expected = np.stack([sosfiltfilt(s, ref, axis=-1).astype(np.float32) for s in filters])
    expected_slow = sosfiltfilt(slow, ref, axis=-1)[..., 30::60].astype(np.float32)
    np.testing.assert_allclose(output, expected, atol=4e-5, rtol=2e-5)
    np.testing.assert_allclose(slow_output, expected_slow, atol=4e-5, rtol=2e-5)
