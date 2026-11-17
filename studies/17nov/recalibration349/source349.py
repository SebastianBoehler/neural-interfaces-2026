"""Exact frozen347B/C models, source-only inputs and the actual ONEBN consumer."""
import inspect
import json
import sys
from pathlib import Path
import numpy as np
import torch

STUDY=Path('/mnt/lustre/work/gehler/gfh527/ni26_crops347_full_v1_20261117')
SOURCE='ead1aefb6e50b3f453dfaaa369ff071693b1449a9f44da376a00a43a072b8a76'
sys.path.insert(0,str(STUDY))
from training347 import source,late_data,sha,digest,state_sha
from common346 import restored,permutations,ORDERS,labels,official
from recipe336 import session_bn


def load():
    if sha(STUDY/'source.sha256')!=SOURCE:raise ValueError('347 scientific source changed')
    for line in (STUDY/'source.sha256').read_text().splitlines():
        pin,name=line.split(maxsplit=1)
        if sha(STUDY/name)!=pin:raise ValueError('347 scientific dependency changed:'+name)
    result=json.loads((STUDY/'result.json').read_text());proof=json.loads((STUDY/'before_labels.json').read_text())
    if not proof['all3arms_all3people_all8orders'] or result['before_labels_sha256']!=sha(STUDY/'before_labels.json'):raise ValueError('complete347proof required')
    d,fit,held,people,y=source();eligible,late,qualification=late_data(fit);models={};records={}
    for arm in ('B','C'):
        folder=STUDY/'output'/arm;r=json.loads((folder/'record.json').read_text())
        if r['source_sha256']!=SOURCE or r['held_labels_opened'] or r['source_rows_sha256']!=digest(fit) or r['checkpoint_sha256']!=sha(folder/'model.pt') or r['prediction_sha256']!=sha(folder/'predictions.npz') or r['checkpoint_sha256']!=result['records'][arm]['checkpoint_sha256']:raise ValueError('347 model/source/input identity')
        model,_=restored(d,fit,held);saved=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
        if model.scale!=saved['scale'] or not np.array_equal(model.G,saved['G']):raise ValueError('source6 normalization changed')
        model.net.load_state_dict(saved['state'],strict=True);model.set_k(20.);model.net.eval()
        if state_sha(model.net)!=r['final_state_sha256']:raise ValueError('frozen347state differs')
        actual=[(name,module) for name,module in model.net.named_modules() if isinstance(module,session_bn.SessionBN)]
        if len(actual)!=1 or actual[0][0]!='spatial_conv.1':raise ValueError('actual ONEBN path differs')
        name,bn=actual[0]
        if bn.C!=288 or bn.m!=32 or bn.k!=20. or bn.causal or bn.eps!=1e-5 or bn.momentum!=.1:raise ValueError('actual ONEBN consumer convention differs')
        if any(isinstance(m,torch.nn.modules.dropout._DropoutNd) for m in model.net.modules()):raise ValueError('dropout invalidates source moment extraction')
        if sha(inspect.getfile(type(model.net)))!='604e690ad1743a8dc9eeb87c0c3065d28619cbd4dbda8c9790e60924cafbe81a':raise ValueError('FBCNet forward/ONEBN owner changed')
        models[arm]=model;records[arm]=r
    return d,fit,people,y,eligible,late,models,records,qualification


def bn(model):return model.net.spatial_conv[1]


def parameter_sha(net):
    return digest(np.concatenate([v.detach().numpy().ravel() for _,v in sorted(net.named_parameters())]))


def install(model,mean,var):
    before={n:v.detach().clone() for n,v in model.net.state_dict().items()};parameters=parameter_sha(model.net)
    if mean.shape!=(288,) or var.shape!=(288,) or not np.isfinite(mean).all() or not np.isfinite(var).all() or (var<0).any():raise ValueError('source-only BN statistics schema')
    with torch.no_grad():bn(model).running_mean.copy_(torch.from_numpy(mean));bn(model).running_var.copy_(torch.from_numpy(var))
    changed=[]
    for name,value in model.net.state_dict().items():
        if not torch.equal(value,before[name]):changed.append(name)
    allowed={'spatial_conv.1.running_mean','spatial_conv.1.running_var'}
    if not changed or set(changed)-allowed or parameter_sha(model.net)!=parameters or bn(model).k!=20.:raise ValueError('intervention changed learned weights/other state or did nothing')
    return dict(changed_buffers=changed,parameter_sha256=parameters,mean_sha256=digest(mean),variance_sha256=digest(var))
