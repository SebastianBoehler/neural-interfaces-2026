"""Prepared R-N-N-R source PCA comparison; original pc_basis, no labels/head fits."""
import ast
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
from scipy.linalg import svd
from sklearn.utils.extmath import svd_flip
from threadpoolctl import threadpool_limits,threadpool_info
import torch
from torch.jit._pickle import restore_type_tag

from pca_fixture import ROOT,sha

OWNER=ROOT/'ni26_js321_20261117_returning2/source/scripts/export_tsem.py'
OWNER_SHA='6b45b1de48df7733aa6edc20eb2957a33911f5a20b5324563943a744b21d181c'


def main():
    if len(sys.argv)!=4:raise ValueError('usage: pca_reference.py COMMON.pt NATIVE_EXE NEW_OUTPUT_DIR')
    source,exe,out=Path(sys.argv[1]),Path(sys.argv[2]).resolve(),Path(sys.argv[3])
    if out.exists():raise ValueError('new comparison output directory required')
    if sha(OWNER)!=OWNER_SHA:raise ValueError('original pc_basis owner changed')
    namespace=dict(np=np,svd=svd,svd_flip=svd_flip)
    functions=[n for n in ast.parse(OWNER.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='pc_basis']
    if len(functions)!=1:raise ValueError('original PCA helper missing')
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(OWNER),'exec'),namespace)
    torch.set_num_threads(6)
    original_sha=sha(source);data=torch.load(source,weights_only=True);z=data['source_Z'].numpy()
    if z.shape!=(2278,44239) or z.dtype!=np.float64:raise ValueError('registered common PCA input differs')
    out.mkdir();seconds=[];result=None;baseline=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    loaded_rss=int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')//1024
    with threadpool_limits(limits=6):
        pools=threadpool_info()
        if any(pool['num_threads']!=6 for pool in pools):raise ValueError('reference six-thread context failed')
        for repetition in range(2):
            started=time.perf_counter();mean,v,s=namespace['pc_basis'](z,256)
            basis=v.astype(np.float32);del v
            seconds.append(time.perf_counter()-started);result=(mean,basis,s)
            if repetition==0:
                subprocess.run([str(exe),str(source),str(out/'native.pt')],check=True)
    reference_peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with torch.serialization.safe_globals([restore_type_tag]):native=torch.load(out/'native.pt',weights_only=True)
    mean,basis,s=result
    errors=dict(mean=float(np.max(np.abs(native['mean'].numpy()-mean))),
        basis=float(np.max(np.abs(native['basis'].numpy()-basis))),
        all_singular_relative=float(np.max(np.abs(native['singular_values'].numpy()-s))/s[0]))
    if errors['mean']>1e-12 or errors['basis']>1e-6 or errors['all_singular_relative']>1e-10:
        raise ValueError('fresh reference PCA parity failed: '+str(errors))
    if sha(source)!=original_sha:raise ValueError('common PCA source mutated')
    report=dict(status='PCA_SOURCE_MATCHED_PASS',reference_seconds=seconds,
        native_seconds=native['fit_seconds'].tolist(),reference_loaded_peak_KiB=baseline,
        reference_loaded_rss_KiB=loaded_rss,reference_fit_peak_KiB=reference_peak,
        native_loaded_rss_KiB=int(native['loaded_rss_KiB']),native_loaded_peak_KiB=int(native['loaded_peak_KiB']),
        native_fit_peak_KiB=int(native['fit_peak_KiB']),errors=errors,threads=6,pools=pools,
        owner_sha256=OWNER_SHA,input_sha256=original_sha,exe_sha256=sha(exe),
        affinity=sorted(os.sched_getaffinity(0)),
        memory_scope='current loaded RSS and whole-process high-water before/after fit; includes any earlier parser peak; not process-tree total',
        accuracy_computed=False,query_labels_opened=False,head_fit=False)
    (out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
