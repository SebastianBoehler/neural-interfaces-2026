"""Exact fixed-source Native reference/features, full51covariance blocks plus328slow."""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from data import PRODUCER,sha,digest,metadata

FULL=Path('/mnt/lustre/work/gehler/gfh527/ni26_s301_native_full_v3_20261116')
TRAIN=Path('/mnt/lustre/work/gehler/gfh527/ni26_s301_native_trainref_v3_20261116')


def native():
    core=FULL/'runs/s301_full_build';train=TRAIN/'runs/s301_trainref'
    for folder in (core,train):
        for line in (folder/'library.sha256').read_text().splitlines():
            pin,name=line.split(maxsplit=1)
            path=Path(name)
            if not path.is_absolute():path=(FULL if folder==core else TRAIN)/path
            if sha(path)!=pin:raise ValueError('native library hash differs')
    os.environ['NATIVE_CORE_LIB']=str(core/'libnative_core.so');os.environ['NATIVE_TRAIN_LIB']=str(train/'libtrainref.so')
    path=PRODUCER/'studies/16nov/followups/native_core/bridge.py'
    spec=importlib.util.spec_from_file_location('native334bridge',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m.Native()


def covariances():
    d,people,fit,query,perms=metadata();rows=np.sort(np.r_[fit,query]);position=np.full(len(people),-1);position[rows]=np.arange(len(rows))
    c=np.empty((len(rows),51,41,41),np.float64);slow=np.empty((len(rows),328),np.float32);hashes={}
    for person in [f'P{i:02d}' for i in range(2,11)]:
        path=PRODUCER/f'runs/s301_native_inputs/{person}.npz';hashes[person]=sha(path)
        with np.load(path,allow_pickle=False) as z:
            m=json.loads(str(z['metadata']));r=z['rows'];cc=z['cov'];ss=z['slow']
        from data import PRODUCER_SHA
        if m['person']!=person or m['source_sha256']!=PRODUCER_SHA or m['held_accuracy_computed'] is not False or digest(r)!=m['rows_sha256'] or not np.array_equal(r,np.flatnonzero(people==person)):
            raise ValueError('covcache provenance/rows')
        if cc.shape!=(len(r),51,41,41) or ss.shape!=(len(r),328):raise ValueError('cachelayout')
        c[position[r]]=cc;slow[position[r]]=ss
    return c,slow,rows,position[fit],position[query],people[rows],d,perms,hashes


def fixed_bank(lib,c,slow,source,device):
    handle=lib.train_reference(device)
    try:
        for i in source:lib.add(handle,c[i],slow[i])
        prior=lib.prior(handle);out=np.empty((len(c),44239),np.float64)
        for i in range(len(c)):out[i]=lib.feature(handle,c[i],slow[i])
    finally:lib.train.nt_destroy(handle)
    if not np.isfinite(out).all():raise ValueError('nonfinite fixedfeatures')
    return out,prior,slow[source].astype(np.float64).mean(0)


def main():
    root=Path(os.environ['GEN_ROOT']);torch.set_num_threads(int(os.environ.get('STATIC_THREADS','6')));start=time.monotonic()
    c,s,rows,source,query,people,d,perms,hashes=covariances();lib=native();device=int(os.environ['STATIC_DEVICE'])
    if device!=1 or not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):raise ValueError('registered fullbuildA100required')
    identities={'source7':source}
    identities.update({f'exclude_{p}':source[people[source]!=p] for p in sorted(set(people[source]))})
    records={};dest=root/'fixed_features';dest.mkdir(exist_ok=False)
    for name,fit in identities.items():
        tick=time.monotonic();features,reference,slowmean=fixed_bank(lib,c,s,fit,device)
        out=dest/f'{name}.npz'
        with out.open('xb') as f:np.savez(f,features=features,rows=rows,source=source,query=query,fit=fit,people=people,stream=d['stream'][rows],reference=reference,slow_mean=slowmean)
        records[name]=dict(file_sha256=sha(out),source_rows_sha256=digest(rows[fit]),reference_sha256=digest(reference),
                           slowmean_sha256=digest(slowmean),fit_people=sorted(set(people[fit])),seconds=time.monotonic()-tick)
        print(name,records[name],flush=True)
    report=dict(source_sha256=sha(root/'source.sha256'),cache_hashes=hashes,rows_sha256=digest(rows),P01_used=False,labels_opened=False,
                records=records,device='CUDA' if device else 'CPU',seconds=time.monotonic()-start,scope='static source-only reference, not causal stream reference')
    with (dest/'manifest.json').open('x') as f:json.dump(report,f,indent=2)
    print('334staticfeatureproduction complete',flush=True)


if __name__=='__main__':main()
