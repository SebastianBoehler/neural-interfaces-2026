"""Actual FIR and real-source compatibility audit; no optimizer or class labels."""
import inspect
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from scipy.signal import convolve
from inputs import source,restored,teacher_setup,features,PILOT,sha,digest,GROUPS,FIT
from interventions import variants

root=Path(os.environ['INFO_ROOT']);started=time.monotonic();torch.set_num_threads(6)
x,rows,people=source();net,scale=restored();before={k:v.clone() for k,v in net.state_dict().items()};teacher,bands=teacher_setup()
with np.load(PILOT/'embeddings.npz') as z:
    if sha(PILOT/'embeddings.npz')!='2aa088c2e5df6fc07f7802adf6cb17042c8c5e2c5176440b07650249a841fa04':raise ValueError('savedsourceCfeatures')
    source=np.isin(people[z['rows']],FIT);sd=z['features'][source].astype(np.float64).std(0)
    position=np.full(len(people),-1,int);position[z['rows']]=np.arange(len(z['rows']));expected=z['features'][position[rows]]
f0=features(net,x,scale)
if np.any(sd<=1e-12) or not np.allclose(f0,expected,atol=1e-5,rtol=1e-5):raise ValueError('frozenpopulation source feature identity')
feature_error=float(np.abs(f0-expected).max())
y0,z0=teacher(x)
# Baseline exact teacher cache identity, using already frozen COMMON3teacher/scaler.
targetroot=Path('/mnt/lustre/work/gehler/gfh527/ni26_pretrain337_targets_v1_20261117')
if sha(targetroot/'teacher.npz')!='b679c40c5234c2f1de14052c73a36c3e754c1f9f98b42a7d1bdca393ae1b8dcc':raise ValueError('teacher identity')
with np.load(targetroot/'teacher.npz') as z:
    pos=np.searchsorted(z['rows'],rows);target=z['covariance'][pos]
if not np.allclose(z0,target,atol=1e-4,rtol=1e-6):raise ValueError('actualraw teacher/cache identity')
teacher_error=float(np.abs(z0-target).max())
with torch.no_grad():base=net.spectral_filtering(torch.from_numpy((x.astype(np.float64)/scale).astype(np.float32))).numpy()
coefficients=[p.detach().numpy().astype(np.float64) for p in net.spectral_filtering.fir_list]
for a in net.spectral_filtering.fir_a_list:
    if not np.array_equal(a.detach().numpy(),np.r_[1.,np.zeros(len(a)-1)]):raise ValueError('FIRdenominator nonidentity')
grid=np.arange(0,60.0001,.125);gains=np.stack([np.abs(np.exp(-2j*np.pi*grid[:,None]*np.arange(len(c))[None]/120)@c) for c in coefficients])
# Independent direct-time convolution checks the ACTUALnativefftfunction path andpadding.
expected=np.stack([convolve((x[0,0]/scale).astype(np.float64),c,mode='same',method='direct') for c in coefficients])
if not np.allclose(base[0,:,0],expected,atol=1e-5,rtol=1e-5):raise ValueError('actualFIR/directconvolution')
results={}
for name,changed in variants(x).items():
    _,target=teacher(changed);f=features(net,changed,scale)
    with torch.no_grad():filtered=net.spectral_filtering(torch.from_numpy((changed.astype(np.float64)/scale).astype(np.float32))).numpy()
    delta=(changed.astype(np.float64)-x)/scale;diff=filtered-base;den=float(np.square(delta).sum());center=float(np.square(delta[...,100:380]).sum())
    record=dict(raw_relative_energy=float(np.square(changed.astype(np.float64)-x).sum()/np.square(x.astype(np.float64)).sum()),
        filtered_delta_gain_all=None if den==0 else float(np.square(diff).sum()/den),
        filtered_delta_gain_center=None if center==0 else float(np.square(diff[...,100:380]).sum()/center),
        feature_standardized_RMS=float(np.sqrt(np.mean(((f-f0)/sd)**2))),
        teacher_standardized_RMS={k:float(np.sqrt(np.mean((target[:,g]-z0[:,g])**2))) for k,g in zip(('diagonal','offdiagonal','slow'),GROUPS)},
        teacher_band_RMS=[float(np.sqrt(np.mean((target[:,b*2583:(b+1)*2583]-z0[:,b*2583:(b+1)*2583])**2))) for b in range(17)])
    results[name]=record
if max(results['CAR']['teacher_standardized_RMS'].values())>1e-4:raise ValueError('CAR teacher invariance')
if any(not torch.equal(v,net.state_dict()[k]) for k,v in before.items()):raise ValueError('inference mutatedcheckpoint')
report=dict(source_sha256=sha(root/'source.sha256'),source_rows=rows.tolist(),source_people=sorted(set(people[rows])),source_X_sha256=digest(x),
    checkpoint_sha256=sha(PILOT/'model.pt'),restored_feature_max_abs=feature_error,
    baseline_teacher_max_abs=teacher_error,
    filter_source_path=inspect.getfile(type(net.spectral_filtering)),filter_source_sha256=sha(inspect.getfile(type(net.spectral_filtering))),
    coefficients_sha256=[digest(c) for c in coefficients],coefficient_lengths=list(map(len,coefficients)),native_bands=net.spectral_filtering.band_filters,
    teacher_bands=bands,frequency_grid_Hz=grid.tolist(),gains=gains.tolist(),interventions=results,
    checkpoints=dict(prefine_native_saved=False,auxiliary_saved=False,existing_postpretext_hash='3981dc104410449adfee1b34aeb84fc82918959d2d55f9470c2b21748acccc57'),
    labels_opened=False,optimizer_run=False,P09_X_used=False,P02P03_inference=False,seconds=time.monotonic()-started)
(root/'audit.json').write_text(json.dumps(report));print(json.dumps({k:v for k,v in report.items() if k not in ('gains','frequency_grid_Hz')}),flush=True)
