"""Real P01/P02 eight-person-prior CPU feature/reference parity; no scores."""
import os
import sys
from pathlib import Path
import numpy as np

ROOT = Path(os.environ['NI26_WS']).resolve(strict=True)
CACHE = Path(os.environ['S307_CACHE']).resolve(strict=True)
sys.path[:0] = [str(ROOT/'experiments'), str(Path(__file__).resolve().parent)]
import features as f
from cpu_crossfit import CPUReference


def main():
    with np.load(ROOT/'runs/s307/crossfit/P01_P02.npz',allow_pickle=False) as z:
        prior,theta,mean,scale = (z[k].copy() for k in ('prior','theta','mean','scale'))
        rows,order,actual = (z[k].copy() for k in ('source_rows','order|within0','features|within0'))
    d = np.load(ROOT/'data/proteus_windows.npz',allow_pickle=False)
    session = np.char.add(np.char.add(d['subject'],'|'),d['session'])
    first = order[session[order] == session[order[0]]][:3]
    if len(first) != 3: raise ValueError('three same-session windows required')
    loc = np.searchsorted(rows,first)
    with np.load(CACHE/'runs/s301_native_inputs/P02.npz',allow_pickle=False) as z:
        cov,slow = z['cov'][loc],z['slow'][loc].astype(np.float64)
    logs = f.logm(cov); accum = np.cumsum(logs,axis=0)
    result = []
    for i in range(len(first)):
        ref = f.expm_sym((2*prior + accum[i])/(3+i))
        tangent = f.whiten_tangent(cov[i],ref).reshape(-1)
        centered = slow[i] - slow[:i+1].sum(axis=0)/(3+i)
        result.append(np.concatenate([tangent,centered]))
    expected = np.stack(result); got = actual[loc]
    replay = CPUReference().causal(np.ascontiguousarray(prior),np.ascontiguousarray(cov[:2]),np.ascontiguousarray(slow[:2],dtype=np.float32))
    if not np.array_equal(replay,got[:2]): raise ValueError('CPU prefix/reset guard failed')
    maximum = float(np.max(np.abs(got-expected)))
    relative = float(np.linalg.norm(got-expected)/np.linalg.norm(expected))
    def prob(x):
        logits = ((x-mean)/scale) @ theta[:,:-1].T + theta[:,-1]
        logits -= logits.max(axis=1,keepdims=True)
        p = np.exp(logits); return p/p.sum(axis=1,keepdims=True)
    p,q = prob(expected),prob(got)
    pdiff = float(np.max(np.abs(p-q)))
    sorted_p = np.sort(p,axis=1); nontied = sorted_p[:,-1] > sorted_p[:,-2]
    changed = int(np.count_nonzero((p.argmax(1) != q.argmax(1)) & nontied))
    if relative > 1e-6 or maximum > 1e-4 or pdiff > 1e-5 or changed:
        raise ValueError(f'CPU feature/probability parity failed {relative=} {maximum=} {pdiff=} {changed=}')
    print(f'CPU real source features pass: {relative=} {maximum=} {pdiff=} {changed=}; no scores',flush=True)


if __name__ == '__main__': main()
