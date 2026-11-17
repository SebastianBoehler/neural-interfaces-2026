"""Tiny actual P04 Native-versus-Python fixed-source parity and CPU cost, no labels."""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from static_features import native,PRODUCER,fixed_bank
from guards import cache_probe
from data import sha,metadata


def main():
    root=Path(os.environ['GEN_ROOT']);torch.set_num_threads(6);start=time.monotonic();d,people,fit,query,perms=metadata()
    path=PRODUCER/'runs/s301_native_inputs/P04.npz'
    with np.load(path,allow_pickle=False) as z:c=z['cov'][:3].copy();slow=z['slow'][:3].copy();rows=z['rows'][:3]
    lib=native();tick=time.monotonic();out,prior,sm=fixed_bank(lib,c,slow,np.arange(3),0);cost=time.monotonic()-tick
    sys.path.insert(0,str(PRODUCER/'experiments'));import features as ft
    ref=ft.expm_sym(prior);expected=np.c_[np.concatenate([ft.whiten_tangent(c[:,b],ref[b]) for b in range(51)],axis=1),slow-sm]
    error=float(np.max(np.abs(out-expected)))
    if not np.allclose(out,expected,atol=1e-4,rtol=1e-6):raise ValueError('staticnative/Pythonparity')
    probe=cache_probe((rows[0],c[0],slow[0]));projection=cost/3*(2*len(fit)+len(query))*8
    result=dict(passed=True,labels_opened=False,P01_used=False,source_sha256=sha(root/'source.sha256'),source_rows=rows.tolist(),
                reference_sha256=sha(path),native_python_max_abs=error,cache_probe=probe,measured_three_rows_seconds=cost,
                projected_all8_reference_seconds=projection,projected_cpu_corehours=projection*6/3600,total_seconds=time.monotonic()-start)
    with (root/'tiny.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
