"""Checked ctypes wrapper for the immutable native inference handle."""

import ctypes

import numpy as np


class NativeHead:
    def __init__(self, library, checkpoint, device):
        self.lib = ctypes.CDLL(str(library))
        self.lib.s308_error.restype = ctypes.c_char_p
        self.lib.s308_load.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.lib.s308_load.restype = ctypes.c_void_p
        self.lib.s308_predict.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                          ctypes.c_int64, ctypes.c_void_p]
        self.lib.s308_predict.restype = ctypes.c_int
        self.lib.s308_free.argtypes = [ctypes.c_void_p]
        self.handle = self.lib.s308_load(str(checkpoint).encode(), device)
        if not self.handle:
            raise RuntimeError(self.lib.s308_error().decode())

    def predict(self, x):
        if not self.handle:
            raise RuntimeError('native model is closed')
        x = np.ascontiguousarray(x, dtype=np.float64)
        if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
            raise ValueError('native input must be a finite nonempty matrix')
        out = np.empty((len(x), 3), dtype=np.float64)
        rc = self.lib.s308_predict(self.handle, x.ctypes.data, len(x), x.shape[1], out.ctypes.data)
        if rc:
            raise RuntimeError(self.lib.s308_error().decode())
        if np.any(out < 0) or not np.allclose(out.sum(axis=1), 1, atol=2e-5):
            raise ValueError('native model returned invalid probabilities')
        return out

    def close(self):
        if self.handle:
            self.lib.s308_free(self.handle)
            self.handle = None
