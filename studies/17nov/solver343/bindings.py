"""Exact historical unscored source matrix and unchanged custom C++ binary."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path('/mnt/lustre/work/gehler/gfh527/ni26_s311_cpu_kernel_v1_20261116')
MATRIX_SHA='a38ec018aaf2f727916fa402bb082659543dcfbf758d44e6ee3f9819e4ef8b24'
LIBRARY_SHA='bf3298eb03bbfdfe32b84629ae27e54b77b7c5f0088886b42b11400564805bb0'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def load():
    if sha(ROOT/'source.sha256')!='fb09442bf275646faf14bdf8da37d48c3226adb1ae141ad8fc5cda77ab7d3312':raise ValueError('original311source manifest')
    report=json.loads((ROOT/'runs/s311_smoke/report.json').read_text());path=ROOT/'runs/s311_smoke/source_matrix.npz';library=ROOT/'runs/build/libheadkernels.so'
    if sha(path)!=MATRIX_SHA or sha(library)!=LIBRARY_SHA or report['matrix_sha256']!=MATRIX_SHA or report['held_accuracy_computed']:raise ValueError('unscoredsource/libraryidentity')
    with np.load(path,allow_pickle=False) as z:x,y,theta=(z[k].copy() for k in ('x','y','theta'))
    if x.shape!=(2465,44239) or x.dtype!=np.float64 or y.dtype!=np.int64 or not x.flags.c_contiguous or not np.array_equal(np.unique(y),[0,1,2]):raise ValueError('source shape/layout')
    if np.ascontiguousarray(x).ctypes.data!=x.ctypes.data or np.ascontiguousarray(y).ctypes.data!=y.ctypes.data:raise ValueError('unexpectedbinding input copy')
    here=ROOT/'studies/16nov/followups/head_kernel';pins=dict(line.split(maxsplit=1)[::-1] for line in (ROOT/'source.sha256').read_text().splitlines())
    key='studies/16nov/followups/head_kernel/kernel.py'
    if sha(here/'kernel.py')!=pins[key]:raise ValueError('originalbindingchanged')
    sys.path.insert(0,str(here));from kernel import Kernel
    return Kernel(library),x,y,theta,report
