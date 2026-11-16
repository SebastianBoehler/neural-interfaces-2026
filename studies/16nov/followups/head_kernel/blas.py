"""Checked explicit-LP64 OpenBLAS CPU experiment interface."""

import ctypes

import numpy as np


class Blas:
    def __init__(self, path):
        self.lib = ctypes.CDLL(str(path))
        for name in ('s311b_error', 's311b_config', 's311b_path'):
            getattr(self.lib, name).restype = ctypes.c_char_p
        self.lib.s311b_threads.restype = ctypes.c_int
        self.lib.s311b_objective.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_void_p, ctypes.c_double, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.s311b_fit.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_double, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        for name in ('s311b_objective', 's311b_fit'):
            getattr(self.lib, name).restype = ctypes.c_int

    def info(self):
        return {'config': self.lib.s311b_config().decode(), 'path': self.lib.s311b_path().decode(),
                'threads': self.lib.s311b_threads()}

    @staticmethod
    def _inputs(x, y):
        x = np.ascontiguousarray(x, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.int64)
        if x.ndim != 2 or not len(x) or y.shape != (len(x),):
            raise ValueError('invalid matrix or labels')
        return x, y

    def objective(self, x, y, theta):
        x, y = self._inputs(x, y)
        theta = np.ascontiguousarray(theta, dtype=np.float64)
        if theta.shape != (3, x.shape[1] + 1):
            raise ValueError('invalid theta shape')
        loss, grad = np.empty(1, np.float64), np.empty_like(theta)
        if self.lib.s311b_objective(x.ctypes.data, y.ctypes.data, *x.shape, theta.ctypes.data,
                                    .3, loss.ctypes.data, grad.ctypes.data):
            raise RuntimeError(self.lib.s311b_error().decode())
        return float(loss[0]), grad

    def fit(self, x, y):
        x, y = self._inputs(x, y)
        theta = np.empty((3, x.shape[1] + 1), dtype=np.float64)
        loss, grad, calls = np.empty(1, np.float64), np.empty(1, np.float64), np.empty(1, np.int64)
        if self.lib.s311b_fit(x.ctypes.data, y.ctypes.data, *x.shape, .3, theta.ctypes.data,
                              loss.ctypes.data, grad.ctypes.data, calls.ctypes.data):
            raise RuntimeError(self.lib.s311b_error().decode())
        return theta, float(loss[0]), float(grad[0]), int(calls[0])
