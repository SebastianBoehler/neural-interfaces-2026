"""Explicit CPU construction for the unchanged native causal-feature kernel."""
import numpy as np
from bridge import Native, pointer


class CpuNative(Native):
    def causal(self, prior, cov, slow):
        h = self.core.nc_create(pointer(prior), 0)
        if not h:
            raise RuntimeError(self.core.nc_error().decode())
        output = np.empty((len(cov), 44239), np.float64)
        try:
            for i in range(len(cov)):
                self.check(self.core.nc_step(h, pointer(cov[i]), pointer(slow[i]), pointer(output[i])), 'nc_error')
        finally:
            self.core.nc_destroy(h)
        return output
