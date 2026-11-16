"""No-score real-EEG head parity gate without a pytest dependency."""
import ctypes
import os
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[4]
PD, PI = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64)
lib = ctypes.CDLL(os.environ['NATIVE_CORE_LIB'])
lib.nc_head_predict.argtypes = [PD, ctypes.c_int64, ctypes.c_int64, PD, PD, PD, PD, PD, PD]
lib.nc_head_objective.argtypes = [PD, PI, ctypes.c_int64, ctypes.c_int64, PD, ctypes.c_double, PD, PD]
lib.nc_head_fit.argtypes = [PD, PI, ctypes.c_int64, ctypes.c_int64, ctypes.c_double, PD, PD, PD, PI]
lib.nc_head_standardize.argtypes = [PD, ctypes.c_int64, ctypes.c_int64, PD, PD, PD]
for name in ('nc_head_predict', 'nc_head_objective', 'nc_head_fit', 'nc_head_standardize'):
    getattr(lib, name).restype = ctypes.c_int


def ptr(x):
    return x.ctypes.data_as(PI if x.dtype == np.int64 else PD)


with np.load(ROOT / 'data/proteus_windows.npz', allow_pickle=False) as z:
    x = np.ascontiguousarray(z['X'][:96, :8].mean(axis=-1), dtype=np.float64)
    y = np.ascontiguousarray(z['y'][:96], dtype=np.int64)
n, d = x.shape
scaler = StandardScaler().fit(x)
standard = np.ascontiguousarray(scaler.transform(x))
model = LogisticRegression(C=.3, max_iter=2000, tol=1e-10).fit(standard, y)
arrays = [np.ascontiguousarray(a, dtype=np.float64) for a in (scaler.mean_, scaler.scale_, model.coef_, model.intercept_)]
native_standard = np.empty_like(x)
assert lib.nc_head_standardize(ptr(x), n, d, ptr(arrays[0]), ptr(arrays[1]), ptr(native_standard)) == 0
np.testing.assert_allclose(native_standard, standard, atol=1e-14, rtol=1e-14)
logits, probs = np.empty((n, 3)), np.empty((n, 3))
assert lib.nc_head_predict(ptr(x), n, d, *map(ptr, arrays), ptr(logits), ptr(probs)) == 0
np.testing.assert_allclose(logits, model.decision_function(standard), atol=1e-12)
np.testing.assert_allclose(probs, model.predict_proba(standard), atol=1e-12)
theta = np.linspace(-.2, .3, 3 * (d + 1)).reshape(3, -1)
loss, grad = np.empty(1), np.empty_like(theta)
assert lib.nc_head_objective(ptr(standard), ptr(y), n, d, ptr(theta), .3, ptr(loss), ptr(grad)) == 0
logit = standard @ theta[:, :-1].T + theta[:, -1]
expected = np.mean(logsumexp(logit, axis=1) - logit[np.arange(n), y]) + np.square(theta[:, :-1]).sum() / (2 * .3 * n)
delta = softmax(logit, axis=1) - np.eye(3)[y]
expected_grad = np.column_stack((delta.T @ standard / n + theta[:, :-1] / (.3 * n), delta.mean(0)))
np.testing.assert_allclose(loss[0], expected, atol=1e-13)
np.testing.assert_allclose(grad, expected_grad, atol=1e-13)
fitted, fit_loss, gradmax, evaluations = np.zeros_like(theta), np.empty(1), np.empty(1), np.empty(1, dtype=np.int64)
assert lib.nc_head_fit(ptr(standard), ptr(y), n, d, .3, ptr(fitted), ptr(fit_loss), ptr(gradmax), ptr(evaluations)) == 0
ref_theta = np.ascontiguousarray(np.column_stack((model.coef_, model.intercept_)))
ref_loss = np.empty(1)
assert lib.nc_head_objective(ptr(standard), ptr(y), n, d, ptr(ref_theta), .3, ptr(ref_loss), ptr(grad)) == 0
assert abs(fit_loss[0] - ref_loss[0]) <= 1e-8 and gradmax[0] <= 1e-6 and 0 < evaluations[0] <= 2500
np.testing.assert_allclose(softmax(standard @ fitted[:, :-1].T + fitted[:, -1], axis=1), model.predict_proba(standard), atol=1e-5)
print(f'head objective/gradient and fit passed: objective difference={abs(fit_loss[0] - ref_loss[0]):.3g}, max gradient={gradmax[0]:.3g}; no score')
