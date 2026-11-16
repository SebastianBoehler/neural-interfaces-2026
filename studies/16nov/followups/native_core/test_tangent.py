"""Causal tangent-state parity, including session reset and future independence."""
import ctypes
import os
from pathlib import Path

import numpy as np


def eigen_apply(x, fn):
    w, v = np.linalg.eigh(x)
    return np.einsum('...ij,...j,...kj->...ik', v, fn(w), v)


def test_native_causal_tangent_reset_and_prefix():
    path = Path(os.environ.get('NATIVE_CORE_LIB', '/nonexistent/native_core.so'))
    assert path.is_file(), 'native tangent library must be built first'
    lib = ctypes.CDLL(str(path))
    pd, pf = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_float)
    lib.nc_create.argtypes, lib.nc_create.restype = [pd, ctypes.c_int], ctypes.c_void_p
    lib.nc_step.argtypes, lib.nc_step.restype = [ctypes.c_void_p, pd, pf, pd], ctypes.c_int
    lib.nc_reset.argtypes, lib.nc_reset.restype = [ctypes.c_void_p], ctypes.c_int
    lib.nc_destroy.argtypes = [ctypes.c_void_p]
    rng = np.random.default_rng(1)
    prior = np.ascontiguousarray(np.zeros((51, 41, 41), dtype=np.float64))
    factors = rng.normal(size=(3, 51, 41, 41)) / 12
    cov = np.ascontiguousarray(factors @ factors.transpose(0, 1, 3, 2) + np.eye(41))
    slow = np.ascontiguousarray(rng.normal(size=(3, 328)), dtype=np.float32)
    logcov = eigen_apply(cov, lambda w: np.log(np.maximum(w, 1e-12)))
    expected = []
    iu = np.triu_indices(41)
    weight = np.where(iu[0] == iu[1], 1., np.sqrt(2.))
    for t in range(3):
        mean = (2 * prior + logcov[:t + 1].sum(0)) / (t + 3)
        reference = eigen_apply(mean, np.exp)
        inv = eigen_apply(reference, lambda w: 1 / np.sqrt(np.maximum(w, 1e-12)))
        tangent = eigen_apply(inv @ cov[t] @ inv, lambda w: np.log(np.maximum(w, 1e-12)))
        slow_ref = slow[t].astype(np.float64) - slow[:t + 1].astype(np.float64).sum(0) / (t + 3)
        expected.append(np.r_[(tangent[:, iu[0], iu[1]] * weight).ravel(), slow_ref])
    for device in map(int, os.environ.get('NATIVE_CORE_TEST_DEVICES', '0').split(',')):
        handle = lib.nc_create(prior.ctypes.data_as(pd), device)
        assert handle, f'native session creation failed on device {device}'
        output = np.empty((3, 44239), np.float64)
        for t in range(3):
            assert lib.nc_step(handle, cov[t].ctypes.data_as(pd), slow[t].ctypes.data_as(pf), output[t].ctypes.data_as(pd)) == 0
        np.testing.assert_allclose(output, np.asarray(expected), atol=1e-8, rtol=1e-8)
        assert lib.nc_reset(handle) == 0
        again = np.empty(44239, np.float64)
        assert lib.nc_step(handle, cov[0].ctypes.data_as(pd), slow[0].ctypes.data_as(pf), again.ctypes.data_as(pd)) == 0
        np.testing.assert_array_equal(again, output[0])
        lib.nc_destroy(handle)
