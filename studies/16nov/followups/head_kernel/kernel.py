"""Typed access to original autograd and independent analytical CPU kernels."""

import ctypes

import numpy as np


class Kernel:
    def __init__(self, path):
        self.lib = ctypes.CDLL(str(path))
        for name in ('nc_head_error', 's311_error'):
            getattr(self.lib, name).restype = ctypes.c_char_p
        obj = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p,
               ctypes.c_double, ctypes.c_void_p, ctypes.c_void_p]
        fit = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_double,
               ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        for name in ('nc_head_objective', 's311_objective'):
            getattr(self.lib, name).argtypes = obj
            getattr(self.lib, name).restype = ctypes.c_int
        for name in ('nc_head_fit', 's311_fit'):
            getattr(self.lib, name).argtypes = fit
            getattr(self.lib, name).restype = ctypes.c_int

    def _inputs(self, x, y):
        x = np.ascontiguousarray(x, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.int64)
        if x.ndim != 2 or not len(x) or y.shape != (len(x),):
            raise ValueError('invalid matrix or labels')
        return x, y

    def objective(self, x, y, theta, analytical):
        x, y = self._inputs(x, y)
        theta = np.ascontiguousarray(theta, dtype=np.float64)
        if theta.shape != (3, x.shape[1] + 1):
            raise ValueError('invalid theta shape')
        loss, grad = np.empty(1, np.float64), np.empty_like(theta)
        name = 's311_objective' if analytical else 'nc_head_objective'
        rc = getattr(self.lib, name)(x.ctypes.data, y.ctypes.data, *x.shape, theta.ctypes.data,
                                     .3, loss.ctypes.data, grad.ctypes.data)
        if rc:
            error = self.lib.s311_error if analytical else self.lib.nc_head_error
            raise RuntimeError(error().decode())
        return float(loss[0]), grad

    def fit(self, x, y, analytical):
        x, y = self._inputs(x, y)
        theta = np.empty((3, x.shape[1] + 1), dtype=np.float64)
        loss, grad, calls = np.empty(1, np.float64), np.empty(1, np.float64), np.empty(1, np.int64)
        name = 's311_fit' if analytical else 'nc_head_fit'
        rc = getattr(self.lib, name)(x.ctypes.data, y.ctypes.data, *x.shape, .3, theta.ctypes.data,
                                     loss.ctypes.data, grad.ctypes.data, calls.ctypes.data)
        if rc:
            error = self.lib.s311_error if analytical else self.lib.nc_head_error
            raise RuntimeError(error().decode())
        return theta, float(loss[0]), float(grad[0]), int(calls[0])
