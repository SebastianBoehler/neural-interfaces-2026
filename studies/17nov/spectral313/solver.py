"""Explicit CPU device of the frozen §309 logistic fitting API."""

import ctypes

import numpy as np


class Solver:
    def __init__(self, path):
        self.lib = ctypes.CDLL(str(path))
        self.lib.nc_head_error.restype = ctypes.c_char_p
        self.lib.nc_head_fit_device.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_double, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p]
        self.lib.nc_head_fit_device.restype = ctypes.c_int

    def fit(self, x, y):
        x = np.ascontiguousarray(x, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.int64)
        if x.ndim != 2 or not len(x) or y.shape != (len(x),) or set(y) != {0, 1, 2}:
            raise ValueError('source fit must contain all three classes')
        theta = np.empty((3, x.shape[1]+1), dtype=np.float64)
        loss, grad, calls = np.empty(1), np.empty(1), np.empty(1, dtype=np.int64)
        rc = self.lib.nc_head_fit_device(x.ctypes.data, y.ctypes.data, *x.shape, .3, 0,
                                         theta.ctypes.data, loss.ctypes.data, grad.ctypes.data, calls.ctypes.data)
        if rc:
            raise RuntimeError(self.lib.nc_head_error().decode())
        if not np.isfinite(theta).all() or not np.isfinite([loss[0], grad[0]]).all() or grad[0] > 1e-5 or calls[0] <= 0:
            raise ValueError('CPU logistic fit is not finite or converged')
        return theta, float(loss[0]), float(grad[0]), int(calls[0])
