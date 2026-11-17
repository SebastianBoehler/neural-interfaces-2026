"""Reference L-BFGS-B with its stopping gradient checked in original coordinates."""
import ctypes
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize


class ReferenceSpan:
    create_symbol = "s343r_create"
    full = False

    def __init__(self, path):
        self.lib = ctypes.CDLL(str(path))
        self.lib.s343r_error.restype = ctypes.c_char_p
        create_types = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                         ctypes.c_int64, ctypes.c_double, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.s343r_create.argtypes = create_types
        self.lib.s343f_create.argtypes = create_types
        self.lib.s343r_destroy.argtypes = [ctypes.c_void_p]
        for name, count in [('objective', 4), ('original_gradient', 3), ('reconstruct', 3)]:
            getattr(self.lib, f's343r_{name}').argtypes = [ctypes.c_void_p]*count
        self.active = None

    def check(self, rc):
        if rc:
            raise RuntimeError(self.lib.s343r_error().decode())

    def fit(self, x, y):
        if x.dtype != np.float64 or y.dtype != np.int64 or not x.flags.c_contiguous or not y.flags.c_contiguous:
            raise ValueError('float64/int64 C contiguous source arrays required')
        if x.ndim != 2 or y.shape != (len(x),):
            raise ValueError('invalid source shape')
        handle, factor = ctypes.c_void_p(), np.empty(7, np.float64)
        self.check(getattr(self.lib, self.create_symbol)(x.ctypes.data, y.ctypes.data, *x.shape, .3,
                                        ctypes.byref(handle), factor.ctypes.data))
        rank = int(factor[1])
        calls, checks, last, stop = 0, 0, {}, {}
        projection_seconds = 0.
        def objective(theta):
            nonlocal calls
            theta = np.ascontiguousarray(theta)
            loss, grad = np.empty(1, np.float64), np.empty_like(theta)
            self.check(self.lib.s343r_objective(handle, theta.ctypes.data, loss.ctypes.data, grad.ctypes.data))
            calls += 1
            last.update(theta=theta.copy(), loss=float(loss[0]), gradient=grad)
            return float(loss[0]), grad
        def accepted(intermediate_result):
            nonlocal checks, projection_seconds
            theta = np.ascontiguousarray(intermediate_result.x)
            if not np.array_equal(theta, last['theta']):
                raise ValueError('accepted iterate differs from last evaluated parameters')
            if self.full:
                checks += 1
                return
            maximum = np.empty(1, np.float64)
            tick = time.perf_counter()
            self.check(self.lib.s343r_original_gradient(handle, last['gradient'].ctypes.data, maximum.ctypes.data))
            projection_seconds += time.perf_counter()-tick
            checks += 1
            if not self.full and maximum[0] <= 1e-5:
                stop.update(reason='original_gradient', maximum=float(maximum[0]))
                raise StopIteration
        try:
            tick = time.perf_counter()
            result = minimize(objective, np.zeros(3*(rank+1)), method='L-BFGS-B', jac=True,
                              callback=accepted, options=dict(maxiter=2000, maxls=50,
                              gtol=1e-5 if self.full else 0., ftol=64*np.finfo(float).eps))
            solve_seconds = time.perf_counter()-tick
            if not stop and not result.success:
                raise RuntimeError(f'reference optimizer did not converge: {result.message}')
            theta = np.empty((3, x.shape[1]+1), np.float64)
            tick = time.perf_counter()
            self.check(self.lib.s343r_reconstruct(handle, result.x.ctypes.data, theta.ctypes.data))
            reconstruction_seconds = time.perf_counter()-tick
            if self.full:
                stats = dict(validation_s=float(factor[0]), input_width=rank)
            else:
                stats = dict(zip(('factor_s', 'rank', 'eigen_min', 'eigen_max', 'cutoff',
                                  'discarded_abs_eigen_sum', 'gram_relative_residual'), map(float, factor)))
            stats.update(solve_s=solve_seconds, reconstruct_s=reconstruction_seconds,
                         original_gradient_projection_s=projection_seconds, accepted_iterations_observed=checks,
                         additional_original_gradient_checks=0 if self.full else checks,
                         iterations=int(result.nit), evaluations=calls, scipy_nfev=int(result.nfev),
                         scipy_njev=int(result.njev), stop=stop or dict(reason=str(result.message)),
                         final_reduced_objective=float(result.fun))
            if result.nfev != calls or checks != result.nit:
                raise ValueError('optimizer evaluation/accepted-iterate counters differ')
            return theta, stats, calls
        finally:
            self.lib.s343r_destroy(handle)


class FullReference(ReferenceSpan):
    create_symbol = "s343f_create"
    full = True
