"""Checked ctypes boundary for native Lane 5 training and session inference."""
import copy
import ctypes as C
from pathlib import Path

import numpy as np

F = C.POINTER(C.c_float)
I64 = C.POINTER(C.c_int64)
I32 = C.POINTER(C.c_int32)
D = C.POINTER(C.c_double)
U64 = C.POINTER(C.c_uint64)
P = C.c_void_p


def raw_array(x):
    a = np.asarray(x, dtype=np.float32)
    if a.ndim != 3 or a.shape[1:] != (41, 480) or not len(a) or not np.isfinite(a).all():
        raise ValueError(f'expected finite, nonempty (n, 41, 480) windows, got {a.shape}')
    return np.ascontiguousarray(a)


def labels_array(y, n):
    a = np.asarray(y)
    if a.shape != (n,) or not np.issubdtype(a.dtype, np.integer) or not np.isin(a, [0, 1, 2]).all():
        raise ValueError('expected n integer labels in {0, 1, 2}')
    return np.ascontiguousarray(a, dtype=np.int64)


def matrix(x, shape):
    a = np.asarray(x, dtype=np.float32)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f'expected finite shape {shape}, got {a.shape}')
    return np.ascontiguousarray(a)


def ptr(a, kind=F):
    return a.ctypes.data_as(kind)


class Native:
    def __init__(self, path):
        self.lib = C.CDLL(str(Path(path).resolve(strict=True)))
        bindings = {
            'n5_error': ([], C.c_char_p), 'n5_create': ([C.c_int, C.c_int, C.c_int64], P),
            'n5_clone': ([P], P), 'n5_destroy': ([P], None), 'n5_features': ([F, C.c_int64, F], C.c_int),
            'n5_set_norm': ([P, F, F], C.c_int), 'n5_param_count': ([P], C.c_int64),
            'n5_get_params': ([P, F], C.c_int), 'n5_set_params': ([P, F, C.c_int64], C.c_int),
            'n5_head_forward': ([P, F, C.c_int64, F], C.c_int),
            'n5_head_grads': ([P, F, I64, F, F, C.c_int64, F, F, F], C.c_int),
            'n5_adam_step': ([P, F, I64, F, F, C.c_int64, C.c_double, C.c_double], C.c_int),
            'n5_predict': ([P, F, C.c_int64, F], C.c_int),
            'n5_fit': ([P, F, I64, I32, C.c_int64, C.c_int32, C.c_int, C.c_int], C.c_int),
            'n5_fit_status': ([P, I64, D, D, U64], C.c_int),
            'n5_consistency_grad': ([P, D], C.c_int),
            'n5_reset': ([P], C.c_int), 'n5_step': ([P, F, C.c_int, F], C.c_int),
            'n5_status': ([P, I64, I64, I64, D, D], C.c_int),
            'n5_save': ([P, C.c_char_p], C.c_int), 'n5_load': ([P, C.c_char_p], C.c_int),
        }
        for name, (args, result) in bindings.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, result

    def check(self, rc):
        if rc != 0:
            message = self.lib.n5_error()
            raise RuntimeError(message.decode(errors='replace') if message else f'native error {rc}')

    def features(self, windows):
        x = raw_array(windows); out = np.empty((len(x), 246), np.float32)
        self.check(self.lib.n5_features(ptr(x), len(x), ptr(out)))
        if not np.isfinite(out).all():
            raise ValueError('native feature output is non-finite')
        return out

    def create(self, arm, device, seed):
        if arm not in (0, 1, 2, 3) or device not in (0, 1):
            raise ValueError('invalid native arm/device')
        handle = self.lib.n5_create(arm, device, seed)
        if not handle:
            self.check(-1)
        return Solver(self, handle)


class Solver:
    def __init__(self, native, handle):
        self.native, self.handle = native, handle

    def _call(self, name, *args):
        if not self.handle:
            raise RuntimeError('native solver is closed')
        self.native.check(getattr(self.native.lib, name)(self.handle, *args))

    def close(self):
        if self.handle:
            self.native.lib.n5_destroy(self.handle)
            self.handle = None

    def __del__(self):
        self.close()

    def clone(self):
        if not self.handle:
            raise RuntimeError('native solver is closed')
        h = self.native.lib.n5_clone(self.handle)
        if not h:
            self.native.check(-1)
        return Solver(self.native, h)

    def __deepcopy__(self, memo):
        return self.clone()

    def set_norm(self, mean, sd):
        mean, sd = matrix(mean, (246,)), matrix(sd, (246,))
        if (sd <= 0).any():
            raise ValueError('normalization scale must be positive')
        self._call('n5_set_norm', ptr(mean), ptr(sd))

    def params(self):
        count = self.native.lib.n5_param_count(self.handle)
        if count <= 0:
            self.native.check(-1)
        out = np.empty(count, dtype=np.float32)
        self._call('n5_get_params', ptr(out))
        return out

    def set_params(self, values):
        a = matrix(values, (len(self.params()),))
        self._call('n5_set_params', ptr(a), len(a))

    def head_forward(self, features):
        a = matrix(features, (len(features), 246)); out = np.empty((len(a), 3), np.float32)
        self._call('n5_head_forward', ptr(a), len(a), ptr(out))
        return out

    def head_grads(self, features, labels, weights, mask):
        x = matrix(features, (len(features), 246)); y = labels_array(labels, len(x))
        w, m = matrix(weights, (len(x),)), matrix(mask, x.shape)
        loss = C.c_float(); dx = np.empty_like(x); dp = np.empty_like(self.params())
        self._call('n5_head_grads', ptr(x), ptr(y, I64), ptr(w), ptr(m), len(x), C.byref(loss), ptr(dx), ptr(dp))
        return float(loss.value), dx, dp

    def adam_step(self, features, labels, weights, mask, lr, wd):
        x = matrix(features, (len(features), 246)); y = labels_array(labels, len(x))
        w, m = matrix(weights, (len(x),)), matrix(mask, x.shape)
        self._call('n5_adam_step', ptr(x), ptr(y, I64), ptr(w), ptr(m), len(x), lr, wd)

    def fit(self, windows, labels, people, held, epochs, consistency):
        x = raw_array(windows); y = labels_array(labels, len(x)); people = np.asarray(people)
        if (people.shape != (len(x),) or not np.issubdtype(people.dtype, np.integer) or
                held in people or len(np.unique(people)) != 9 or epochs < 1 or consistency not in (0, 1)):
            raise ValueError('invalid fit people, held-out person, epochs, or consistency')
        if any(set(y[people == p]) != {0, 1, 2} for p in np.unique(people)):
            raise ValueError('each fitting person must include all three classes')
        people = np.ascontiguousarray(people, dtype=np.int32)
        self._call('n5_fit', ptr(x), ptr(y, I64), ptr(people, I32), len(x), held, epochs, consistency)

    def predict(self, windows):
        x = raw_array(windows); out = np.empty((len(x), 3), dtype=np.float32)
        self._call('n5_predict', ptr(x), len(x), ptr(out))
        if not np.isfinite(out).all():
            raise ValueError('non-finite native logits')
        return out

    def reset(self):
        self._call('n5_reset')

    def step(self, window, update):
        x = raw_array(np.asarray(window)[None]); out = np.empty(3, dtype=np.float32)
        self._call('n5_step', ptr(x), int(bool(update)), ptr(out))
        if not np.isfinite(out).all():
            raise ValueError('non-finite native session logits')
        return out

    def status(self):
        a, b, c = C.c_int64(), C.c_int64(), C.c_int64()
        g, d = C.c_double(), C.c_double()
        self._call('n5_status', C.byref(a), C.byref(b), C.byref(c), C.byref(g), C.byref(d))
        return {'windows': a.value, 'updates': b.value, 'prefix_count': c.value,
                'max_grad': g.value, 'delta': d.value}

    def fit_status(self):
        steps, loss, grad, digest = C.c_int64(), C.c_double(), C.c_double(), C.c_uint64()
        self._call('n5_fit_status', C.byref(steps), C.byref(loss), C.byref(grad), C.byref(digest))
        consistency_grad = C.c_double()
        self._call('n5_consistency_grad', C.byref(consistency_grad))
        return {'steps': steps.value, 'consistency_sum': loss.value,
                'projection_grad_max': grad.value, 'shuffle_hash': digest.value,
                'consistency_grad_max': consistency_grad.value}

    def save(self, path):
        p = Path(path)
        if p.exists():
            raise FileExistsError(p)
        self._call('n5_save', str(p).encode())

    def load(self, path):
        self._call('n5_load', str(Path(path).resolve(strict=True)).encode())


class Session:
    def __init__(self, population):
        self.solver = population.clone()
        self.solver.reset()

    def step(self, current_window, update):
        return self.solver.step(current_window, update)

    def reset(self):
        self.solver.reset()

    def __deepcopy__(self, memo):
        result = object.__new__(Session)
        result.solver = copy.deepcopy(self.solver, memo)
        return result
