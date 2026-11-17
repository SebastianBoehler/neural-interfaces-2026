"""Typed native source-span fit; no alternate solver or device fallback."""
import ctypes
import numpy as np


class Span:
    def __init__(self, path):
        self.lib = ctypes.CDLL(str(path))
        self.lib.s343_error.restype = ctypes.c_char_p
        self.lib.s343_fit.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                     ctypes.c_int64, ctypes.c_double, ctypes.c_void_p,
                                     ctypes.c_void_p, ctypes.c_void_p]
        self.lib.s343_fit.restype = ctypes.c_int

    def fit(self, x, y):
        if x.dtype != np.float64 or y.dtype != np.int64 or not x.flags.c_contiguous or not y.flags.c_contiguous:
            raise ValueError('fit requires float64/int64 C contiguous inputs')
        if x.ndim != 2 or y.shape != (len(x),):
            raise ValueError('invalid input shapes')
        theta = np.empty((3, x.shape[1] + 1), np.float64)
        stats, calls = np.empty(14, np.float64), np.empty(1, np.int64)
        rc = self.lib.s343_fit(x.ctypes.data, y.ctypes.data, *x.shape, .3,
                                theta.ctypes.data, stats.ctypes.data, calls.ctypes.data)
        if rc:
            raise RuntimeError(self.lib.s343_error().decode())
        names = ('factor_s', 'solve_s', 'reconstruct_s', 'fullcheck_s', 'rank', 'eigen_min',
                 'eigen_max', 'cutoff', 'discarded_abs_eigen_sum', 'objective', 'max_gradient',
                 'reduced_objective', 'reduced_max_gradient', 'gram_relative_residual')
        return theta, dict(zip(names, map(float, stats))), int(calls[0])
