"""C-aware pinned GPU head with an independent NumPy loss/gradient."""
import ctypes
import time

import numpy as np
import torch
from scipy.special import logsumexp, softmax

from common import sha

LIBRARY = '/mnt/lustre/work/gehler/gfh527/ni26_s309_device_20261116/runs/s309_build/libhead_device.so'
LIB_SHA = '027df0434ed7096b6110d0c500bbd83c796a473d21426f3953dbc245b11b394e'
PD = ctypes.POINTER(ctypes.c_double)
PI = ctypes.POINTER(ctypes.c_int64)


def objective(x, y, theta, c):
    w, b = theta[:, :-1], theta[:, -1]
    logits = x @ w.T + b
    norm = logsumexp(logits, axis=1)
    loss = np.mean(norm - logits[np.arange(len(y)), y]) + np.sum(w*w)/(2*c*len(y))
    residual = np.exp(logits - norm[:, None])
    residual[np.arange(len(y)), y] -= 1
    gradient = np.column_stack((residual.T @ x/len(y) + w/(c*len(y)), residual.mean(0)))
    return float(loss), gradient


class Head:
    def __init__(self):
        if sha(LIBRARY) != LIB_SHA:
            raise ValueError('GPU-proven library changed')
        self.lib = ctypes.CDLL(LIBRARY)
        self.lib.nc_head_fit_device.argtypes = [PD, PI, ctypes.c_int64, ctypes.c_int64,
                                               ctypes.c_double, ctypes.c_int, PD, PD, PD, PI]
        self.lib.nc_head_error.restype = ctypes.c_char_p

    def fit(self, x, y, c, device=1):
        if x.dtype != np.float64 or not x.flags.c_contiguous or y.dtype != np.int64 or not y.flags.c_contiguous:
            raise ValueError('F64C/I64C inputs required')
        if c <= 0 or set(y) != {0, 1, 2} or not np.isfinite(x).all():
            raise ValueError('invalid head inputs')
        theta = np.empty((3, x.shape[1]+1), np.float64)
        loss, grad, evaluations = ctypes.c_double(), ctypes.c_double(), ctypes.c_int64()
        start = time.monotonic()
        status = self.lib.nc_head_fit_device(x.ctypes.data_as(PD), y.ctypes.data_as(PI), len(y),
                 x.shape[1], c, device, theta.ctypes.data_as(PD), ctypes.byref(loss),
                 ctypes.byref(grad), ctypes.byref(evaluations))
        if status:
            raise RuntimeError(self.lib.nc_head_error().decode())
        check, gradient = objective(x, y, theta, c)
        maximum = float(np.abs(gradient).max())
        if not np.isfinite(theta).all() or not np.any(theta[:, :-1]) or abs(check-loss.value)>1e-9 or maximum>1e-5:
            raise ValueError('independent objective/convergence/activity guard failed')
        if evaluations.value >= 2500:
            raise ValueError('evaluation cap reached')
        return theta, dict(C=c, n=len(y), d=x.shape[1], loss=check, max_gradient=maximum,
                          evaluations=evaluations.value, reported_gradient=grad.value,
                          seconds=time.monotonic()-start, device=device)


def predict(x, theta):
    p = softmax(x @ theta[:, :-1].T + theta[:, -1], axis=1)
    if not np.isfinite(p).all() or not np.allclose(p.sum(1), 1):
        raise ValueError('invalid posterior')
    return p
