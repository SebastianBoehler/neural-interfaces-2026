"""Use the same native causal kernel on CPU for an isolated §307 cross-fit."""
import importlib.util
import os
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ['NI26_WS']).resolve(strict=True)
sys.path.insert(0,str(ROOT/'studies/16nov/followups/native_core'))
from bridge import Native, pointer


class CPUReference(Native):
    def causal(self, prior, cov, slow):
        handle = self.core.nc_create(pointer(prior), 0)
        if not handle: raise RuntimeError(self.core.nc_error().decode())
        output = np.empty((len(cov),44239),np.float64)
        try:
            for i in range(len(cov)):
                self.check(self.core.nc_step(handle,pointer(cov[i]),pointer(slow[i]),pointer(output[i])),'nc_error')
        finally:
            self.core.nc_destroy(handle)
        return output


spec = importlib.util.spec_from_file_location('cpu_crossfit_base',HERE/'crossfit.py')
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
base.Native = CPUReference
if __name__ == '__main__':
    base.main(sys.argv[1],sys.argv[2],len(sys.argv)==4 and sys.argv[3]=='smoke')
