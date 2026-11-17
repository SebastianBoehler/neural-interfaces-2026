"""Selected real source EEG only; exact fixed teacher and restored337 native model."""
import importlib.util
import json
import os
import sys
import zipfile
from pathlib import Path
import numpy as np
import torch

DEP=Path('/mnt/lustre/work/gehler/gfh527/ni26_cov337_opt_v1_20261117')
sys.path.insert(0,str(DEP));from dependencies import sha,digest,metadata,PRODUCER,BASE
from targets import GROUPS
sys.path.insert(0,'/mnt/lustre/work/gehler/gfh527/ni26_reference_bridge336_v1_20261117')
from recipe336 import deep,session_bn,state_sha
PILOT=Path('/mnt/lustre/work/gehler/gfh527/ni26_pretrain337_pilot_v3_20261117/output/covariance')
FIXTURE=Path('/mnt/lustre/work/gehler/gfh527/ni26_cov337_prepare_v2_20261117/teacher_fixture.npz')
FIT=('P04','P05','P06','P07','P08','P10')


def source():
    d,people,fit,query,perms=metadata();rows=np.sort(np.r_[tuple(np.flatnonzero(people==p)[:2] for p in FIT)])
    if set(people[rows])!=set(FIT) or len(rows)!=12:raise ValueError('source audit selection')
    data=BASE/'data/proteus_windows.npz';out=[]
    with zipfile.ZipFile(data) as archive,archive.open('X.npy') as f:
        version=np.lib.format.read_magic(f)
        shape,fortran,dtype=np.lib.format.read_array_header_1_0(f) if version==(1,0) else np.lib.format.read_array_header_2_0(f)
        if shape!=(2655,41,480) or fortran or dtype!=np.dtype('float32'):raise ValueError('raw schema')
        start=f.tell();size=41*480*dtype.itemsize
        for row in rows:
            f.seek(start+int(row)*size);out.append(np.frombuffer(f.read(size),dtype=dtype).reshape(41,480).copy())
    return np.stack(out),rows,people


def restored():
    path=PILOT/'model.pt'
    if sha(path)!='cdfb792fa6213e77a8fedf23a539a51f3ab15f1f59ec4d9a90ed8f1cc6645401':raise ValueError('337final C checkpoint')
    saved=torch.load(path,map_location='cpu',weights_only=False);torch.manual_seed(0);net=deep.build('fbcnet',41,480)
    session_bn.swap_bn(net,False,20.);net.load_state_dict(saved['state'],strict=True)
    for module in net.modules():
        if isinstance(module,session_bn.SessionBN):module.k=None
    net.eval()
    if state_sha(net)!='4ab0cef0c9c67de14e6952d68ac56d6b6b090e5fa187efb9e3d2b9f88ebb5f9b':raise ValueError('exact restored native state')
    filt=net.spectral_filtering
    if filt.method!='fir' or filt.phase!='zero' or any(p.requires_grad for p in filt.parameters()):raise ValueError('actualnativefilterchanged')
    return net,saved['scale']


def teacher_setup():
    if sha(FIXTURE)!='06644bae5a331022c755184ad71089b08c8e079271917e4a4223eafa8bd3d187':raise ValueError('COMMON3fixedfixture')
    with np.load(FIXTURE) as z:reference=z['reference'];mean=z['mean'];std=z['std'];slowmean=z['slow_mean']
    if sha(PRODUCER/'experiments/features.py')!=sha(BASE/'experiments/features.py'):raise ValueError('teacher/native feature helper mismatch')
    os.environ['NI26_ROOT']=str(PRODUCER)
    spec=importlib.util.spec_from_file_location('teacher340',PRODUCER/'experiments/ts_core2.py');core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    cfg=core.parse('car_ov17_fs2_t0.1_le_k2');R=core.FT.expm_sym(reference)
    def transform(raw):
        car=(raw-raw.mean(1,keepdims=True)).astype(np.float32)
        blocks=core.blocks(car,cfg)
        vector=np.c_[np.concatenate([core.FT.whiten_tangent(c,R[b]) for b,c in enumerate(blocks)],axis=1),core.slow_waveform(car,cutoff=1.).astype(np.float64)-slowmean]
        return vector,(vector-mean)/std
    return transform,core.BANDSETS['ov17']


def features(net,raw,scale):
    values=[]
    def hook(module,args):values.append(args[0].detach().numpy().copy())
    handle=net.final_layer.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():net(torch.from_numpy((raw.astype(np.float64)/scale).astype(np.float32)))
    finally:handle.remove()
    return np.concatenate(values)
