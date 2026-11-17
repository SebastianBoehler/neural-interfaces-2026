"""Common-three-person full-dimensional teacher and real source-only batch fixture."""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from dependencies import metadata,PRODUCER,PRODUCER_SHA,sha,digest,native
from coverage import plan
from targets import scaler,GROUPS

FIT=('P04','P05','P06','P07','P08','P10');COMMON=FIT[:3]
root=Path(os.environ['COV_ROOT']);started=time.monotonic();torch.set_num_threads(2)
d,people,fitrows,queryrows,perms=metadata();rows=[];cov=[];slow=[];hashes={}
for person in FIT:
    path=PRODUCER/f'runs/s301_native_inputs/{person}.npz';hashes[person]=sha(path)
    with np.load(path,allow_pickle=False) as z:r=z['rows'];c=z['cov'];s=z['slow'];m=json.loads(str(z['metadata']))
    if m['person']!=person or m['source_sha256']!=PRODUCER_SHA or m['held_accuracy_computed'] is not False or digest(r)!=m['rows_sha256'] or not np.array_equal(r,np.flatnonzero(people==person)):raise ValueError('source cache provenance')
    rows.append(r);cov.append(c);slow.append(s)
rows=np.concatenate(rows);cov=np.concatenate(cov);slow=np.concatenate(slow)
if not set(people[rows])==set(FIT) or np.intersect1d(rows,queryrows).size:raise ValueError('forbidden rows')
common=np.flatnonzero(np.isin(people[rows],COMMON));plans={}
for label,names in (('coverage3',COMMON),('coverage6',FIT)):
    plans[label]=plan(d,rows[np.isin(people[rows],names)])
    if set(people[plans[label].ravel()])!=set(names):raise ValueError('coverage missingperson')
lib=native();handle=lib.train_reference(0);tick=time.monotonic()
try:
    for i in common:lib.add(handle,cov[i],slow[i])
    reference=lib.prior(handle);source=np.stack([lib.feature(handle,cov[i],slow[i]) for i in common])
    mean,std=scaler(source);selected=np.unique(np.concatenate([p[:2].ravel() for p in plans.values()]))
    order=np.argsort(rows);positions=np.searchsorted(rows[order],selected)
    if np.any(positions>=len(rows)):raise ValueError('fixture rows absent')
    where=order[positions]
    if not np.array_equal(rows[where],selected):raise ValueError('fixture rowmap')
    targets=np.stack([lib.feature(handle,cov[i],slow[i]) for i in where])
finally:lib.train.nt_destroy(handle)
teacher_seconds=time.monotonic()-tick;slowmean=slow[common].astype(np.float64).mean(0)
sys.path.insert(0,str(PRODUCER/'experiments'));import features as ft
chosen=where[:3];ref=ft.expm_sym(reference)
expected=np.c_[np.concatenate([ft.whiten_tangent(cov[chosen,b],ref[b]) for b in range(51)],1),slow[chosen]-slowmean]
error=float(np.max(np.abs(expected-targets[:3])))
if not np.allclose(expected,targets[:3],atol=1e-4,rtol=1e-6):raise ValueError('independent fullteacher parity')
with (root/'teacher_fixture.npz').open('xb') as out:np.savez(out,rows=selected,targets=targets,mean=mean,std=std,
    reference=reference,slow_mean=slowmean,common_rows=rows[common],**plans)
report=dict(source_sha256=sha(root/'source.sha256'),fit_people=FIT,common_teacher_people=COMMON,held='P09',
    common_rows_sha256=digest(rows[common]),reference_sha256=digest(reference),scaler_mean_sha256=digest(mean),scaler_std_sha256=digest(std),
    fixture_sha256=sha(root/'teacher_fixture.npz'),fixture_rows_sha256=digest(selected),cache_hashes=hashes,
    native_python_max_abs=error,group_sizes=list(map(len,GROUPS)),teacher_seconds=teacher_seconds,seconds=time.monotonic()-started,
    full_teacher_source_rows=len(common),fixture_rows=len(selected),equal_exposures={k:int(p.size) for k,p in plans.items()},
    labels_opened=False,model_optimization=False,P02P03_inference=False,source_quality_score=False,passed=True)
(root/'prepare.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
