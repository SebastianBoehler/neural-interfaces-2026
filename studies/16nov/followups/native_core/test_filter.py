"""Check the native per-window zero-phase filter against SciPy."""
import ctypes
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import butter, sosfilt_zi, sosfiltfilt

HERE = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    source = HERE / "filter.cpp"
    assert source.is_file(), "native SOS filter implementation is missing"
    library = tmp_path_factory.mktemp("native_core") / "filter.so"
    subprocess.run(["c++", "-std=c++17", "-O2", "-shared", "-fPIC", str(source), "-o", str(library)], check=True)
    lib = ctypes.CDLL(str(library))
    ptr = ctypes.POINTER(ctypes.c_double)
    lib.nc_sosfiltfilt.argtypes = [ptr, ctypes.c_int, ptr, ptr, ctypes.c_int, ctypes.c_int, ptr]
    lib.nc_sosfiltfilt.restype = ctypes.c_int
    return lib.nc_sosfiltfilt


@pytest.mark.parametrize("kind", ["band", "low"])
def test_window_filter_matches_scipy(native, kind):
    sos = butter(4, [1, 3] if kind == "band" else 1, btype=kind, fs=120, output="sos")
    zi = np.ascontiguousarray(sosfilt_zi(sos), dtype=np.float64)
    sos = np.ascontiguousarray(sos, dtype=np.float64)
    x = np.ascontiguousarray(np.random.default_rng(4).normal(size=480), dtype=np.float64)
    out = np.empty_like(x)
    ptr = ctypes.POINTER(ctypes.c_double)
    call = lambda a: a.ctypes.data_as(ptr)
    pad = 3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum()))
    assert native(call(x), len(x), call(sos), call(zi), len(sos), pad, call(out)) == 0
    np.testing.assert_allclose(out, sosfiltfilt(sos, x), atol=2e-11, rtol=2e-11)
    np.testing.assert_array_equal(out.astype(np.float32), sosfiltfilt(sos, x).astype(np.float32))
