"""Typed bindings for the §301 native preprocessing, reference and head APIs."""
import ctypes
import os

import numpy as np

PD = ctypes.POINTER(ctypes.c_double)
PF = ctypes.POINTER(ctypes.c_float)
PI = ctypes.POINTER(ctypes.c_int64)


def pointer(x):
    if not x.flags.c_contiguous or x.dtype not in (np.float32, np.float64, np.int64):
        raise ValueError('native input must be contiguous float32, float64 or int64')
    return x.ctypes.data_as({np.dtype('float32'): PF, np.dtype('float64'): PD, np.dtype('int64'): PI}[x.dtype])


class Native:
    def __init__(self):
        self.core = ctypes.CDLL(os.environ['NATIVE_CORE_LIB'])
        self.train = ctypes.CDLL(os.environ['NATIVE_TRAIN_LIB'])
        c, t = self.core, self.train
        c.nc_preprocess.argtypes = [PF, PD, PD, PD, PD, PF, PF]
        c.nc_covariances.argtypes = [PF, PD]
        c.nc_create.argtypes, c.nc_create.restype = [PD, ctypes.c_int], ctypes.c_void_p
        c.nc_step.argtypes = [ctypes.c_void_p, PD, PF, PD]
        c.nc_destroy.argtypes = [ctypes.c_void_p]
        c.nc_head_predict.argtypes = [PD, ctypes.c_int64, ctypes.c_int64, PD, PD, PD, PD, PD, PD]
        c.nc_head_fit.argtypes = [PD, PI, ctypes.c_int64, ctypes.c_int64, ctypes.c_double, PD, PD, PD, PI]
        c.nc_head_objective.argtypes = [PD, PI, ctypes.c_int64, ctypes.c_int64, PD, ctypes.c_double, PD, PD]
        c.nc_head_standardize.argtypes = [PD, ctypes.c_int64, ctypes.c_int64, PD, PD, PD]
        t.nt_create.argtypes, t.nt_create.restype = [ctypes.c_int], ctypes.c_void_p
        t.nt_add.argtypes = [ctypes.c_void_p, PD, PF]
        t.nt_prior.argtypes = [ctypes.c_void_p, PD]
        t.nt_feature.argtypes = [ctypes.c_void_p, PD, PF, PD]
        t.nt_combine.argtypes = [PD, PI, ctypes.c_int, PD]
        t.nt_destroy.argtypes = [ctypes.c_void_p]
        c.nc_error.restype, c.nc_head_error.restype, t.nt_error.restype = ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p

    def check(self, result, source):
        if result:
            raise RuntimeError(getattr(self.train if source == 'nt_error' else self.core, source)().decode())

    def preprocess(self, x, coefficients):
        if x.shape != (41, 480) or x.dtype != np.float32:
            raise ValueError('expected one float32 41×480 EEG window')
        band = np.empty((17, 41, 480), np.float32)
        slow = np.empty(328, np.float32)
        cov = np.empty((51, 41, 41), np.float64)
        self.check(self.core.nc_preprocess(pointer(x), *map(pointer, coefficients), pointer(band), pointer(slow)), 'nc_error')
        self.check(self.core.nc_covariances(pointer(band), pointer(cov)), 'nc_error')
        return cov, slow

    def train_reference(self, device):
        h = self.train.nt_create(device)
        if not h:
            raise RuntimeError(self.train.nt_error().decode())
        return h

    def add(self, h, cov, slow):
        self.check(self.train.nt_add(h, pointer(cov), pointer(slow)), 'nt_error')

    def prior(self, h):
        out = np.empty((51, 41, 41), np.float64)
        self.check(self.train.nt_prior(h, pointer(out)), 'nt_error')
        return out

    def feature(self, h, cov, slow):
        out = np.empty(44239, np.float64)
        self.check(self.train.nt_feature(h, pointer(cov), pointer(slow), pointer(out)), 'nt_error')
        return out

    def combine(self, priors, counts, held):
        out = np.empty((51, 41, 41), np.float64)
        self.check(self.train.nt_combine(pointer(priors), pointer(counts), held, pointer(out)), 'nt_error')
        return out

    def standardize(self, x, mean, scale):
        out = np.empty_like(x)
        self.check(self.core.nc_head_standardize(pointer(x), len(x), x.shape[1], pointer(mean), pointer(scale), pointer(out)), 'nc_head_error')
        return out

    def fit(self, x, y):
        theta = np.empty((3, x.shape[1] + 1), np.float64)
        loss, grad = np.empty(1, np.float64), np.empty(1, np.float64)
        calls = np.empty(1, np.int64)
        self.check(self.core.nc_head_fit(pointer(x), pointer(y), len(x), x.shape[1], .3,
                                        pointer(theta), pointer(loss), pointer(grad), pointer(calls)), 'nc_head_error')
        return theta, float(loss[0]), float(grad[0]), int(calls[0])

    def objective(self, x, y, theta):
        loss, grad = np.empty(1, np.float64), np.empty_like(theta)
        self.check(self.core.nc_head_objective(pointer(x), pointer(y), len(x), x.shape[1], pointer(theta),
                                              .3, pointer(loss), pointer(grad)), 'nc_head_error')
        return float(loss[0]), grad

    def predict(self, x, mean, scale, coef, intercept):
        logits, probabilities = np.empty((len(x), 3), np.float64), np.empty((len(x), 3), np.float64)
        self.check(self.core.nc_head_predict(pointer(x), len(x), x.shape[1], pointer(mean), pointer(scale),
                                            pointer(coef), pointer(intercept), pointer(logits), pointer(probabilities)), 'nc_head_error')
        return probabilities

    def causal(self, prior, cov, slow):
        h = self.core.nc_create(pointer(prior), 1)
        if not h:
            raise RuntimeError(self.core.nc_error().decode())
        output = np.empty((len(cov), 44239), np.float64)
        try:
            for i in range(len(cov)):
                self.check(self.core.nc_step(h, pointer(cov[i]), pointer(slow[i]), pointer(output[i])), 'nc_error')
        finally:
            self.core.nc_destroy(h)
        return output
