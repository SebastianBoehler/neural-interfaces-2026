"""Training-session reference and population-prior parity."""
import ctypes
import os
from pathlib import Path

import numpy as np

from test_tangent import eigen_apply


def test_training_reference_and_features():
    path = Path(os.environ.get('NATIVE_TRAIN_LIB', '/nonexistent/native_train.so'))
    assert path.is_file(), 'native training-reference library must be built first'
    lib = ctypes.CDLL(str(path))
    pd, pf = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_float)
    lib.nt_create.argtypes, lib.nt_create.restype = [ctypes.c_int], ctypes.c_void_p
    lib.nt_add.argtypes = [ctypes.c_void_p, pd, pf]
    lib.nt_prior.argtypes = [ctypes.c_void_p, pd]
    lib.nt_feature.argtypes = [ctypes.c_void_p, pd, pf, pd]
    lib.nt_destroy.argtypes = [ctypes.c_void_p]
    rng = np.random.default_rng(2)
    factors = rng.normal(size=(3, 51, 41, 41)) / 12
    cov = np.ascontiguousarray(factors @ factors.transpose(0, 1, 3, 2) + np.eye(41))
    slow = np.ascontiguousarray(rng.normal(size=(3, 328)), dtype=np.float32)
    logcov = eigen_apply(cov, lambda w: np.log(np.maximum(w, 1e-12)))
    prior = logcov.mean(0)
    ref = eigen_apply(prior, np.exp)
    inv = eigen_apply(ref, lambda w: 1 / np.sqrt(np.maximum(w, 1e-12)))
    iu = np.triu_indices(41)
    weight = np.where(iu[0] == iu[1], 1., np.sqrt(2.))
    for device in map(int, os.environ.get('NATIVE_CORE_TEST_DEVICES', '0').split(',')):
        h = lib.nt_create(device)
        assert h, 'training reference creation failed'
        for i in range(3):
            assert lib.nt_add(h, cov[i].ctypes.data_as(pd), slow[i].ctypes.data_as(pf)) == 0
        actual_prior = np.empty_like(prior)
        assert lib.nt_prior(h, actual_prior.ctypes.data_as(pd)) == 0
        np.testing.assert_allclose(actual_prior, prior, atol=1e-10, rtol=1e-10)
        for i in range(3):
            actual = np.empty(44239, np.float64)
            assert lib.nt_feature(h, cov[i].ctypes.data_as(pd), slow[i].ctypes.data_as(pf), actual.ctypes.data_as(pd)) == 0
            tangent = eigen_apply(inv @ cov[i] @ inv, lambda w: np.log(np.maximum(w, 1e-12)))
            expected = np.r_[(tangent[:, iu[0], iu[1]] * weight).ravel(),
                             slow[i].astype(np.float64) - slow.astype(np.float64).mean(0)]
            np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-8)
        lib.nt_destroy(h)


def test_fold_prior_excludes_held_person():
    path = Path(os.environ.get('NATIVE_TRAIN_LIB', '/nonexistent/native_train.so'))
    assert path.is_file(), 'native training-reference library must be built first'
    lib = ctypes.CDLL(str(path))
    pd, pi = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64)
    lib.nt_combine.argtypes = [pd, pi, ctypes.c_int, pd]
    rng = np.random.default_rng(7)
    means = np.ascontiguousarray(rng.normal(size=(10, 51, 41, 41)))
    counts = np.ascontiguousarray(np.arange(1, 11), dtype=np.int64)
    result = np.empty_like(means[0])
    assert lib.nt_combine(means.ctypes.data_as(pd), counts.ctypes.data_as(pi), 3, result.ctypes.data_as(pd)) == 0
    keep = np.arange(10) != 3
    np.testing.assert_allclose(result, np.average(means[keep], axis=0, weights=counts[keep]), atol=1e-13)
    means[3] = 1e10
    again = np.empty_like(result)
    assert lib.nt_combine(means.ctypes.data_as(pd), counts.ctypes.data_as(pi), 3, again.ctypes.data_as(pd)) == 0
    np.testing.assert_array_equal(result, again)
