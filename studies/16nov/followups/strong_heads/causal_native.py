"""Explicit CPU/GPU choice for session-local native strong-core features."""

import numpy as np
from bridge import Native, pointer


class CausalNative(Native):
    def __init__(self, device):
        if device not in (0, 1):
            raise ValueError('device must be explicit CPU 0 or GPU 1')
        super().__init__()
        self.device = device

    def causal(self, prior, cov, slow):
        h = self.core.nc_create(pointer(prior), self.device)
        if not h:
            raise RuntimeError(self.core.nc_error().decode())
        output = np.empty((len(cov), 44239), np.float64)
        try:
            for i in range(len(cov)):
                self.check(self.core.nc_step(h, pointer(cov[i]), pointer(slow[i]), pointer(output[i])), 'nc_error')
        finally:
            self.core.nc_destroy(h)
        return output
