"""Real-window 51-block covariance parity against the reference estimator."""
import ctypes
import subprocess
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

from test_preprocess import BANDS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def test_covariance_blocks_real_window(tmp_path):
    source = HERE / 'covariance.cpp'
    assert source.is_file(), 'native covariance implementation is missing'
    library = tmp_path / 'covariance.so'
    subprocess.run(['c++', '-std=c++17', '-O2', '-shared', '-fPIC', str(source), '-o', str(library)], check=True)
    lib = ctypes.CDLL(str(library))
    pf, pd = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_double)
    lib.nc_covariances.argtypes = [pf, pd]
    lib.nc_covariances.restype = ctypes.c_int
    with np.load(ROOT / 'data/proteus_windows.npz', allow_pickle=False) as z:
        x = z['X'][0]
    x = (x - x.mean(axis=0, keepdims=True)).astype(np.float32)
    filtered = np.ascontiguousarray([sosfiltfilt(butter(4, list(b), btype='band', fs=120, output='sos'), x).astype(np.float32) for b in BANDS])
    result = np.empty((51, 41, 41), dtype=np.float64)
    assert lib.nc_covariances(filtered.ctypes.data_as(pf), result.ctypes.data_as(pd)) == 0
    reference = []
    for b in filtered:
        for block in (b, b[:, :240], b[:, 240:]):
            centered = block - block.mean(-1, keepdims=True)
            cov = np.einsum('ct,dt->cd', centered, centered, dtype=np.float64) / block.shape[-1]
            cov = .9 * cov + .1 * np.trace(cov) / 41 * np.eye(41)
            reference.append(cov)
    np.testing.assert_allclose(result, np.asarray(reference), atol=1e-6, rtol=1e-6)
