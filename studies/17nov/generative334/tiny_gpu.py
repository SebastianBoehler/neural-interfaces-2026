"""Same-source CPU/A100 static-feature parity and warm preparation cost only."""
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from static_features import native,fixed_bank,PRODUCER
from data import sha,digest,metadata


def main():
    root=Path(os.environ['GEN_ROOT']);torch.set_num_threads(6)
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):raise ValueError('A100only')
    d,people,source,query,perms=metadata();path=PRODUCER/'runs/s301_native_inputs/P04.npz'
    with np.load(path) as z:c=z['cov'][:3].copy();slow=z['slow'][:3].copy()
    lib=native();cpu= fixed_bank(lib,c,slow,np.arange(3),0)[0]
    costs=[];errors=[]
    for _ in range(3):
        torch.cuda.synchronize();start=time.monotonic();gpu,prior,sm=fixed_bank(lib,c,slow,np.arange(3),1)
        torch.cuda.synchronize();costs.append(time.monotonic()-start);errors.append(float(np.max(np.abs(cpu-gpu))))
        if not np.allclose(cpu,gpu,atol=1e-4,rtol=1e-6):raise ValueError('CPU/A100fullfeatureparity')
    projection=max(costs[1:])/3*(2*len(source)+len(query))*8
    result=dict(passed=True,labels_opened=False,P01_used=False,source_sha256=sha(root/'source.sha256'),
                covariance_cache_sha256=sha(path),reference_sha256=digest(prior),slowmean_sha256=digest(sm),
                costs_seconds=costs,CPU_A100_maxabs=errors,projected_all8_seconds=projection,
                projected_A100hours=projection/3600,projected_CPUcorehours=projection*6/3600,
                limits='tiny warmed preparation, not wholepilot/worker runtime')
    with (root/'tiny_gpu.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
