"""Checked session-local repair ABI over the unchanged §293 population handle."""
import ctypes as C
from pathlib import Path

import numpy as np

F = C.POINTER(C.c_float)
I64 = C.POINTER(C.c_int64)
D = C.POINTER(C.c_double)
P = C.c_void_p


class RepairNative:
    def __init__(self, path):
        self.lib = C.CDLL(str(Path(path).resolve(strict=True)))
        bindings = {
            'n5r_error': ([], C.c_char_p), 'n5r_abi': ([], C.c_int),
            'n5r_create': ([P, C.c_int], P), 'n5r_destroy': ([P], None),
            'n5r_reset': ([P], C.c_int), 'n5r_step': ([P, F, F], C.c_int),
            'n5r_teacher': ([P, F, F], C.c_int),
            'n5r_status': ([P, I64, I64, I64, I64, I64, D, D], C.c_int),
            'n5r_selections': ([P, I64, C.c_int64], C.c_int),
        }
        for name, (args, result) in bindings.items():
            fn = getattr(self.lib, name); fn.argtypes = args; fn.restype = result
        if self.lib.n5r_abi() != 1:
            raise RuntimeError('unexpected repair ABI')

    def check(self, code):
        if code:
            raise RuntimeError(self.lib.n5r_error().decode(errors='replace'))

    def create(self, population, mode):
        if mode not in (1, 2) or not population.handle:
            raise ValueError('expected fresh population handle and fixed repair mode 1 or 2')
        h = self.lib.n5r_create(population.handle, mode)
        if not h:
            self.check(-1)
        return RepairSession(self, population, h)


class RepairSession:
    def __init__(self, native, population, handle):
        self.native, self.population, self.handle = native, population, handle

    def close(self):
        if self.handle:
            self.native.lib.n5r_destroy(self.handle); self.handle = None

    def __del__(self):
        self.close()

    def reset(self):
        self.native.check(self.native.lib.n5r_reset(self.handle))

    def call(self, name, window):
        x = np.asarray(window, dtype=np.float32)
        if x.shape != (41, 480) or not np.isfinite(x).all():
            raise ValueError('expected one finite (41, 480) window')
        x = np.ascontiguousarray(x); out = np.empty(3, np.float32)
        self.native.check(getattr(self.native.lib, name)(self.handle, x.ctypes.data_as(F), out.ctypes.data_as(F)))
        if not np.isfinite(out).all():
            raise ValueError('nonfinite repair logits')
        return out

    def step(self, window):
        return self.call('n5r_step', window)

    def teacher(self, window):
        return self.call('n5r_teacher', window)

    def status(self):
        counts = [C.c_int64() for _ in range(5)]; values = [C.c_double() for _ in range(2)]
        self.native.check(self.native.lib.n5r_status(self.handle, *(C.byref(x) for x in counts + values)))
        result = dict(zip(('windows', 'attempts', 'updates', 'skips', 'selected_total', 'max_grad', 'delta'),
                          [x.value for x in counts + values]))
        selected = np.empty(result['attempts'], dtype=np.int64)
        self.native.check(self.native.lib.n5r_selections(self.handle, selected.ctypes.data_as(I64), len(selected)))
        result['selections'] = selected.tolist()
        return result
