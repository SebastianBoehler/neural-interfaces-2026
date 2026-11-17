"""Head-only source-B1 regression/timing, no model score or raw-window core claim."""
import ctypes
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from bindings import load,sha,ROOT

root=Path(os.environ['SOLVER_ROOT']);started=time.monotonic();old,x,y,theta,metadata=load()
patch=Path('/mnt/lustre/work/gehler/gfh527/ni26_solver341_validation_bench_v1_20261117')
passed=json.loads((patch/'validation_result.json').read_text())
if not passed['exact_fitted_parameters'] or passed['full_parameter_max_abs']!=0:raise ValueError('validation-first')
import sys
sys.path.insert(0,str(ROOT/'studies/16nov/followups/head_kernel'));from kernel import Kernel
library=patch/'build/libheadvalidation.so'
if sha(library)!=passed['new_library_sha256']:raise ValueError('newlibraryhash')
new=Kernel(library);reference=Path('/mnt/lustre/work/gehler/gfh527/ni26_s301_native_reference_all_20261116/runs/s301_reference_all/P01.npz')
if sha(reference)!=metadata['reference_sha256']:raise ValueError('source-referencepin')
with np.load(reference) as z:mean=z['mean'].copy();scale=z['scale'].copy()
raw=np.ascontiguousarray(x[:16]*scale+mean);weights=np.ascontiguousarray(theta[:,:-1]);bias=np.ascontiguousarray(theta[:,-1])
if not np.allclose((raw-mean)/scale,x[:16],atol=1e-12,rtol=1e-12):raise ValueError('real-source rawfeature roundtrip')
for k in (old,new):
    k.lib.nc_head_predict.argtypes=[ctypes.c_void_p,ctypes.c_int64,ctypes.c_int64]+[ctypes.c_void_p]*6
    k.lib.nc_head_predict.restype=ctypes.c_int

def call(lib,row):
    logits=np.empty((1,3));p=np.empty((1,3))
    result=lib.lib.nc_head_predict(row.ctypes.data,1,row.shape[1],mean.ctypes.data,scale.ctypes.data,weights.ctypes.data,bias.ctypes.data,logits.ctypes.data,p.ctypes.data)
    if result:raise RuntimeError(lib.lib.nc_head_error().decode())
    return logits,p

pairs=[];maximum=0.;torch.set_num_threads(1)
with threadpool_limits(limits=1):
    for lib in (old,new):
        for i in range(8):call(lib,raw[i:i+1])
    for repeat in range(12):
        index=repeat%16;row=raw[index:index+1];times={};outputs={}
        for name in (('old','new') if repeat%2==0 else ('new','old')):
            tick=time.perf_counter();outputs[name]=call({'old':old,'new':new}[name],row);times[name]=time.perf_counter()-tick
        difference=max(float(np.abs(a-b).max()) for a,b in zip(outputs['old'],outputs['new']));maximum=max(maximum,difference)
        expected=x[index:index+1]@weights.T+bias
        if difference>1e-12 or not np.allclose(outputs['old'][0],expected,atol=1e-9,rtol=1e-9) or not np.isfinite(outputs['new'][1]).all() or not np.allclose(outputs['new'][1].sum(),1):raise ValueError('B1posterior/logitparity')
        pairs.append(dict(repeat=repeat,source_row_offset=index,old_seconds=times['old'],new_seconds=times['new'],ratio=times['new']/times['old']))
report=dict(source_sha256=sha(root/'source.sha256'),matrix_sha256=metadata['matrix_sha256'],reference_sha256=metadata['reference_sha256'],old_library_sha256=metadata['library_sha256'],new_library_sha256=sha(library),
    pairs=pairs,posterior_logit_max_abs=maximum,median_old_ms=float(np.median([p['old_seconds'] for p in pairs])*1000),median_new_ms=float(np.median([p['new_seconds'] for p in pairs])*1000),
    no_accuracy=True,threads=1,affinity=sorted(os.sched_getaffinity(0)),node=os.uname().nodename,seconds=time.monotonic()-started,
    scope='HEADONLY B1 onsourcefeatures;fullrawto-posterior/core/chain/workerNOTremeasured')
(root/'head_latency.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
