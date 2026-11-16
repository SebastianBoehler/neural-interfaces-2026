"""Detect wrong scaling, class layout, L2 convention and native optimizer drift."""
import ctypes
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.special import logsumexp, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PD = ctypes.POINTER(ctypes.c_double)
PI = ctypes.POINTER(ctypes.c_int64)


@pytest.fixture(scope='module')
def head(tmp_path_factory):
    source = HERE / 'head.cpp'
    assert source.is_file(), 'native matched logistic head is missing'
    import torch
    from torch.utils.cpp_extension import include_paths, library_paths
    target = tmp_path_factory.mktemp('native_head') / 'head.so'
    command = ['c++', '-std=c++20', '-O1', '-shared', '-fPIC', str(source), '-o', str(target)]
    command += [f'-I{p}' for p in include_paths()]
    command += [f'-L{p}' for p in library_paths()]
    command += [f'-Wl,-rpath,{p}' for p in library_paths()]
    command += ['-ltorch', '-ltorch_cpu', '-lc10']
    subprocess.run(command, check=True)
    library = ctypes.CDLL(str(target))
    library.nc_head_predict.argtypes = [PD, ctypes.c_int64, ctypes.c_int64, PD, PD, PD, PD, PD, PD]
    library.nc_head_objective.argtypes = [PD, PI, ctypes.c_int64, ctypes.c_int64, PD, ctypes.c_double, PD, PD]
    library.nc_head_fit.argtypes = [PD, PI, ctypes.c_int64, ctypes.c_int64, ctypes.c_double, PD, PD, PD, PI]
    for name in ('nc_head_predict', 'nc_head_objective', 'nc_head_fit'):
        getattr(library, name).restype = ctypes.c_int
    return library


def ptr(x):
    return x.ctypes.data_as(PI if x.dtype == np.int64 else PD)


def fixture():
    # Real EEG channel means; no model accuracy is scored by these checks.
    with np.load(ROOT / 'data/proteus_windows.npz', allow_pickle=False) as z:
        x = z['X'][:96, :8].mean(axis=-1).astype(np.float64)
        y = np.ascontiguousarray(z['y'][:96], dtype=np.int64)
    return np.ascontiguousarray(x), y


def test_fixed_head_matches_exported_scaler_logits_and_class_probabilities(head):
    x, y = fixture()
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=.3, max_iter=2000).fit(scaler.transform(x), y)
    arrays = [np.ascontiguousarray(a, dtype=np.float64) for a in
              (scaler.mean_, scaler.scale_, model.coef_, model.intercept_)]
    logits = np.empty((len(x), 3)); probabilities = np.empty_like(logits)
    assert head.nc_head_predict(ptr(x), len(x), x.shape[1], *map(ptr, arrays),
                                ptr(logits), ptr(probabilities)) == 0
    np.testing.assert_allclose(logits, model.decision_function(scaler.transform(x)), atol=1e-12)
    np.testing.assert_allclose(probabilities, model.predict_proba(scaler.transform(x)), atol=1e-12)


def test_native_training_scaler_matches_the_reference_training_rows(head):
    assert hasattr(head, 'nc_head_standardize'), 'native training scaler is missing'
    head.nc_head_standardize.argtypes = [PD, ctypes.c_int64, ctypes.c_int64, PD, PD, PD]
    head.nc_head_standardize.restype = ctypes.c_int
    x, _ = fixture(); scaler = StandardScaler().fit(x)
    mean, scale = np.ascontiguousarray(scaler.mean_), np.ascontiguousarray(scaler.scale_)
    output = np.empty_like(x)
    assert head.nc_head_standardize(ptr(x), len(x), x.shape[1], ptr(mean), ptr(scale), ptr(output)) == 0
    np.testing.assert_allclose(output, scaler.transform(x), atol=1e-13, rtol=1e-13)


def test_objective_has_mean_cross_entropy_and_unpenalized_intercept(head):
    x, y = fixture(); x = np.ascontiguousarray(StandardScaler().fit_transform(x))
    theta = np.linspace(-.2, .3, 3 * (x.shape[1] + 1)).reshape(3, -1)
    loss = np.empty(1); gradient = np.empty_like(theta)
    assert head.nc_head_objective(ptr(x), ptr(y), len(x), x.shape[1], ptr(theta), .3,
                                  ptr(loss), ptr(gradient)) == 0
    logits = x @ theta[:, :-1].T + theta[:, -1]
    expected = np.mean(logsumexp(logits, axis=1) - logits[np.arange(len(x)), y])
    expected += np.square(theta[:, :-1]).sum() / (2 * .3 * len(x))
    np.testing.assert_allclose(loss[0], expected, atol=1e-13)
    delta = softmax(logits, axis=1) - np.eye(3)[y]
    expected_gradient = np.column_stack((delta.T @ x / len(x) + theta[:, :-1] / (.3 * len(x)),
                                         delta.mean(axis=0)))
    np.testing.assert_allclose(gradient, expected_gradient, atol=1e-13)


def test_native_fit_reaches_the_same_regularized_logistic_objective(head):
    x, y = fixture(); x = np.ascontiguousarray(StandardScaler().fit_transform(x))
    theta = np.zeros((3, x.shape[1] + 1)); loss = np.empty(1); gradmax = np.empty(1)
    iterations = np.empty(1, dtype=np.int64)
    assert head.nc_head_fit(ptr(x), ptr(y), len(x), x.shape[1], .3, ptr(theta),
                            ptr(loss), ptr(gradmax), ptr(iterations)) == 0
    reference = LogisticRegression(C=.3, max_iter=2000, tol=1e-10).fit(x, y)
    ref_theta = np.column_stack((reference.coef_, reference.intercept_))
    ref_loss = np.empty(1); grad = np.empty_like(theta)
    assert head.nc_head_objective(ptr(x), ptr(y), len(x), x.shape[1], ptr(ref_theta),
                                  .3, ptr(ref_loss), ptr(grad)) == 0
    assert abs(loss[0] - ref_loss[0]) <= 1e-8
    assert gradmax[0] <= 1e-6 and 0 < iterations[0] <= 2500
    np.testing.assert_allclose(softmax(x @ theta[:, :-1].T + theta[:, -1], axis=1),
                               reference.predict_proba(x), atol=1e-5)


def test_device_selected_fit_preserves_cpu_solver_and_rejects_unknown_device(head):
    assert hasattr(head, 'nc_head_fit_device'), 'device-resident native fit is missing'
    head.nc_head_fit_device.argtypes = [PD, PI, ctypes.c_int64, ctypes.c_int64,
                                        ctypes.c_double, ctypes.c_int, PD, PD, PD, PI]
    head.nc_head_fit_device.restype = ctypes.c_int
    x, y = fixture(); x = np.ascontiguousarray(StandardScaler().fit_transform(x))
    expected = np.empty((3, x.shape[1] + 1)); actual = np.empty_like(expected)
    loss = np.empty(1); gradient = np.empty(1); evaluations = np.empty(1, dtype=np.int64)
    assert head.nc_head_fit(ptr(x), ptr(y), len(x), x.shape[1], .3, ptr(expected),
                            ptr(loss), ptr(gradient), ptr(evaluations)) == 0
    assert head.nc_head_fit_device(ptr(x), ptr(y), len(x), x.shape[1], .3, 0, ptr(actual),
                                   ptr(loss), ptr(gradient), ptr(evaluations)) == 0
    np.testing.assert_array_equal(actual, expected)
    assert head.nc_head_fit_device(ptr(x), ptr(y), len(x), x.shape[1], .3, 2, ptr(actual),
                                   ptr(loss), ptr(gradient), ptr(evaluations)) != 0
