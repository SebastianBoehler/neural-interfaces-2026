"""Required competition adapter. All model computation runs in libni26_runtime."""
import ctypes
from pathlib import Path
import torch
from benchmark_utils.base_solver import CompetSolver


class NativeModel:
    def __init__(self, directory, device, channels):
        self.directory = Path(directory)
        self.lib = ctypes.CDLL(str(self.directory/'libni26_runtime.so'))
        self.lib.ni26_error.restype = ctypes.c_char_p
        self.lib.ni26_create.argtypes = [ctypes.c_char_p]*3
        self.lib.ni26_create.restype = ctypes.c_void_p
        self.lib.ni26_clone.argtypes = [ctypes.c_void_p]
        self.lib.ni26_clone.restype = ctypes.c_void_p
        self.lib.ni26_destroy.argtypes = [ctypes.c_void_p]
        self.lib.ni26_reset.argtypes = [ctypes.c_void_p]
        self.lib.ni26_predict.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                         ctypes.c_int, ctypes.c_void_p]
        self.handle = self.lib.ni26_create(str(self.directory).encode(), str(device).encode(),
                                           ','.join(channels or ()).encode())
        if not self.handle:
            self._check(-1)

    def _check(self, result):
        if result:
            raise RuntimeError(self.lib.ni26_error().decode())

    def __deepcopy__(self, memo):
        other = object.__new__(NativeModel)
        other.directory, other.lib = self.directory, self.lib
        other.handle = self.lib.ni26_clone(self.handle)
        if not other.handle:
            self._check(-1)
        memo[id(self)] = other
        return other

    def reset_state(self):
        self._check(self.lib.ni26_reset(self.handle))

    def predict(self, windows):
        if (not isinstance(windows, torch.Tensor) or windows.dtype != torch.float32
                or windows.ndim != 3 or tuple(windows.shape[1:]) != (41, 480)
                or not windows.is_contiguous()):
            raise ValueError('native adapter requires contiguous float32[B,41,480] tensor')
        labels = torch.empty(windows.shape[0], dtype=torch.int64)
        self._check(self.lib.ni26_predict(self.handle, windows.data_ptr(), windows.shape[0],
                                         int(windows.is_cuda), labels.data_ptr()))
        return labels

    def __del__(self):
        handle = getattr(self, 'handle', None)
        if handle:
            self.lib.ni26_destroy(handle)
            self.handle = None


class Solver(CompetSolver):
    name = 'NI26Native'

    def load_model(self, meta):
        return NativeModel(meta['submission_dir'], meta['device'], meta.get('ch_names'))
