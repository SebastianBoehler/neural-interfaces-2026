"""Independent probability identities and real-data full-information/session guards."""
import copy
import json
import sys
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from inference import Session,stream,banks,density_stream,ARMS
from model import inverse,transform
from data import DATA,PRODUCER,sha


def cache_probe(probe):
    row,cov,slow=probe
    sys.path.insert(0,str(PRODUCER/'experiments'))
    import ts_core2 as t
    import zipfile
    with zipfile.ZipFile(DATA) as archive,archive.open('X.npy') as f:
        version=np.lib.format.read_magic(f)
        if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(f)
        else:shape,fortran,dtype=np.lib.format.read_array_header_2_0(f)
        if shape!=(2655,41,480) or fortran or dtype!=np.dtype('float32'):raise ValueError('raw window schema')
        start=f.tell();size=41*480*dtype.itemsize;f.seek(start+int(row)*size)
        raw=np.frombuffer(f.read(size),dtype=dtype).reshape(1,41,480).copy()
    raw=(raw-raw.mean(1,keepdims=True)).astype(np.float32)
    actual=np.stack(t.blocks(raw,t.parse('car_ov17_fs2_t0.1_le_k2')),axis=1)[0]
    actual_slow=t.slow_waveform(raw,cutoff=1.)[0]
    err=float(np.max(np.abs(actual-cov)));serr=float(np.max(np.abs(actual_slow-slow)))
    if not np.allclose(actual,cov,atol=1e-4,rtol=1e-6) or serr>1e-4:raise ValueError('independent cache reconstruction differs')
    return dict(covariance_max_abs=err,slow_max_abs=serr)


def check(model,x,eta):
    z=transform(model,x);restored=inverse(model,z)
    inv=float(np.max(np.abs(restored-x)))
    if not np.allclose(restored,x,atol=1e-8,rtol=1e-8):raise ValueError('inverse information loss')
    a=z@model['q'];r=z-a@model['q'].T
    if not np.allclose(r+a@model['q'].T,z,atol=1e-10):raise ValueError('view decomposition lost information')
    cached=banks(model,z,eta);out={};evidence={}
    for arm in ARMS:
        p,state=density_stream(model,cached[arm],eta,arm);direct,_=stream(model,z,eta,arm)
        if not np.allclose(p,direct,atol=1e-9,rtol=1e-9):raise ValueError('cached/direct likelihood mismatch')
        short,_=stream(model,z[:5],eta,arm)
        if not np.array_equal(direct[:5],short):raise ValueError('prefix future influence')
        q=Session(model,eta,arm);q.step(z[0]);cloned=copy.deepcopy(q)
        if not np.array_equal(q.step(z[1]),cloned.step(z[1])):raise ValueError('deepcopy continuation')
        q.reset();again=np.stack([q.step(v) for v in z])
        if not np.array_equal(direct,again):raise ValueError('resetidentity')
        stream(model,z[::-1],eta,arm)
        repeat,_=stream(model,z,eta,arm)
        if not np.array_equal(direct,repeat):raise ValueError('A-B-A session contamination')
        reversed_state=stream(model,z[::-1],eta,arm)[1]
        if not np.allclose(state.evidence,reversed_state.evidence,atol=1e-8,rtol=1e-8):raise ValueError('set posterior order property')
        if not np.isfinite(state.evidence).all():raise ValueError('posterior finite')
        out[arm]=p;evidence[arm]=state.evidence
    if not np.allclose(out['anchor_off'],1/3,atol=1e-10,rtol=0):raise ValueError('uniform permutation symmetry fails')
    if not np.allclose(out['cyclic'],out['mixture'][:,[2,0,1]],atol=1e-9,rtol=1e-9):raise ValueError('cyclic anchor identity')
    if np.max(np.abs(out['mixture']-1/3))<1e-5:raise ValueError('semantic anchor inactive')
    if np.ptp(evidence['mixture'])<1e-6:raise ValueError('hypothesis evidence inactive')
    altered=stream(model,z,.25 if eta!=.25 else 4.,'mixture')[0]
    if np.max(np.abs(altered-out['mixture']))<1e-7:raise ValueError('precision inactive')
    # Independent direct Gaussian density including the otherwise-cancelled ||r||² term.
    state=Session(model,eta,'mixture');one=z[0];aa=one@state.q;rr=one-aa@state.q.T
    expected=-eta/2*((rr[None,None]-state.mean)**2).sum(-1)/(len(one)-2)
    anchor=-eta/4*((aa[None,None]-state.ba[:,None]-state.anchor[None])**2).sum(-1)
    full=anchor[:,None]+expected[:,state.perms]
    joint=np.exp(full-logsumexp(full));manual=joint.sum((0,1))
    if not np.allclose(manual,out['mixture'][0],atol=1e-10):raise ValueError('independent Gaussian marginalization')
    return dict(inverse_max_abs=inv,anchor_max_nonuniform=float(np.max(np.abs(out['mixture']-1/3))),
                final_logweight_range=float(np.ptp(evidence['mixture'])),all_arms=list(ARMS),dimensions=z.shape[1])


def cpu_feature_readout(model,bank,eta):
    from data import PRODUCER
    with np.load(PRODUCER/'runs/s301_native_inputs/P04.npz') as f:
        cov=f['cov'][:17];slow=f['slow'][:17];rows=f['rows'][:17]
    from feature_io import load
    position=np.searchsorted(bank['rows'],rows)
    sys.path.insert(0,str(PRODUCER/'experiments'));import features as ft
    reference=ft.expm_sym(bank['reference'])
    cpu=np.c_[np.concatenate([ft.whiten_tangent(cov[:,b],reference[b]) for b in range(51)],axis=1),slow-bank['slow_mean']]
    existing=bank['features'][position]
    if not np.allclose(cpu,existing,atol=1e-4,rtol=1e-6):raise ValueError('CPU/staticproducerfeatureparity')
    a=stream(model,transform(model,cpu),eta,'mixture')[0]
    b=stream(model,transform(model,existing),eta,'mixture')[0]
    error=float(np.max(np.abs(a-b)))
    margin=np.sort(a,axis=1)[:,-1]-np.sort(a,axis=1)[:,-2]
    changes=int(np.sum((a.argmax(1)!=b.argmax(1))&(margin>1e-6)))
    if error>1e-3 or changes:raise ValueError('actualfittedreadoutCPU/producerparity')
    return dict(feature_max_abs=float(np.max(np.abs(cpu-existing))),fitted_probability_max_abs=error,nontied_argmax_changes=changes)
