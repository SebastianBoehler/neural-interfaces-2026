"""Common-three teacher/statistics unchanged; allsource6+P09 per-window targets."""
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from common import source,PRODUCER,PRODUCER_SHA,sha,digest,native_teacher,FIXTURE,FIXTURE_SHA,COMMON,FIT,moments,plan

root=Path(os.environ['PRETRAIN_ROOT']);started=time.monotonic();torch.set_num_threads(2)
d,fit,held,people,y=source() # Only six source labels; held labels stayclosed.
if sha(FIXTURE)!=FIXTURE_SHA:raise ValueError('common3 teacher fixture changed')
with np.load(FIXTURE) as f:reference=f['reference'];mean=f['mean'];std=f['std'];slowmean=f['slow_mean'];commonrows=f['common_rows']
rows=np.sort(np.r_[fit,held]);cov=np.empty((len(rows),51,41,41),np.float64);slow=np.empty((len(rows),328),np.float32);hashes={}
for person in (*FIT,'P09'):
    path=PRODUCER/f'runs/s301_native_inputs/{person}.npz';hashes[person]=sha(path)
    with np.load(path,allow_pickle=False) as z:r=z['rows'];m=json.loads(str(z['metadata']));c=z['cov'];s=z['slow']
    pos=np.searchsorted(rows,r)
    if m['person']!=person or m['source_sha256']!=PRODUCER_SHA or digest(r)!=m['rows_sha256'] or not np.array_equal(rows[pos],r):raise ValueError('cache rows/provenance')
    cov[pos]=c;slow[pos]=s
common=np.searchsorted(rows,commonrows);lib=native_teacher();handle=lib.train_reference(0)
try:
    for i in common:lib.add(handle,cov[i],slow[i])
    if digest(lib.prior(handle))!=digest(reference):raise ValueError('common3 reference changed')
    targets=np.stack([lib.feature(handle,c,s) for c,s in zip(cov,slow)])
finally:lib.train.nt_destroy(handle)
if not np.allclose(targets[common].mean(0),mean,atol=1e-10) or not np.allclose(targets[common].std(0),std,atol=1e-10):raise ValueError('common3 scaler changed')
mt=moments(torch.from_numpy(d['X'][rows]));mm=mt[common].mean(0);ms=mt[common].std(0)
if np.any(ms<=1e-12):raise ValueError('moment constant coordinate')
trace=plan(d,fit,steps=64);rng=np.random.default_rng(337);order=rng.permutation(len(fit));shuffle=np.empty(len(fit),int);shuffle[order]=np.roll(order,1)
if np.any(shuffle==np.arange(len(fit))) or not np.array_equal(np.sort(shuffle),np.arange(len(fit))):raise ValueError('shuffled target notderangement')
with (root/'teacher.npz').open('xb') as out:np.savez(out,rows=rows,fit=fit,held=held,covariance=((targets-mean)/std).astype(np.float32),
    moment=((mt-mm)/ms).astype(np.float32),trace=trace,shuffle_rows=fit[shuffle],reference=reference,mean=mean,std=std,moment_mean=mm,moment_std=ms)
report=dict(source_sha256=sha(root/'source.sha256'),teacher_sha256=sha(root/'teacher.npz'),fixture_sha256=FIXTURE_SHA,
    fit_rows_sha256=digest(fit),held_rows_sha256=digest(held),trace_sha256=digest(trace),shuffle_rows_sha256=digest(fit[shuffle]),
    common_rows_sha256=digest(commonrows),reference_sha256=digest(reference),scaler_mean_sha256=digest(mean),scaler_std_sha256=digest(std),
    cache_hashes=hashes,source_people=FIT,common_people=COMMON,held='P09',held_labels_opened=False,P02P03_inference=False,seconds=time.monotonic()-started)
(root/'teacher.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
