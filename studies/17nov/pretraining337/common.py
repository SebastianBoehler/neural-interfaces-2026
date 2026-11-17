"""Pinned native source6 recipe; no exposed-person inference or labels."""
import os
import sys
import types
from pathlib import Path
import numpy as np
import torch

API=Path('/mnt/lustre/work/gehler/gfh527/ni26_reference_bridge336_v1_20261117')
PREP=Path('/mnt/lustre/work/gehler/gfh527/ni26_cov337_opt_v1_20261117')
sys.path.insert(0,str(PREP));from dependencies import sha,digest,PRODUCER,PRODUCER_SHA,native as native_teacher
if sha(API/'source.sha256')!='7fa3da6d63270d9911de7d302c1f84e954ab02f0797818a6ae08ebca3b857383':raise ValueError('nativeAPIpin')
sys.path.insert(0,str(API));from recipe336 import native,load,labels,state_sha,observe,inference,official
sys.path.insert(0,str(PREP));from targets import GROUPS
from coverage import plan
FIT=('P04','P05','P06','P07','P08','P10');COMMON=FIT[:3]
FIXTURE=Path('/mnt/lustre/work/gehler/gfh527/ni26_cov337_prepare_v2_20261117/teacher_fixture.npz')
FIXTURE_SHA='06644bae5a331022c755184ad71089b08c8e079271917e4a4223eafa8bd3d187'
ARMS=('moment','covariance','shuffled','extra_supervised')


def source():
    d,allfit,_,people=load();fit=allfit[np.isin(people[allfit],FIT)];held=np.flatnonzero(people=='P09')
    y=labels(fit,len(people));d['y']=np.full(len(people),-1,np.int64);d['y'][fit]=y
    d.update(w=np.ones(len(people)),nolow=np.zeros(len(people),bool),tag='')
    if set(people[fit])!=set(FIT) or np.any(d['y'][held]!=-1):raise ValueError('source6 boundary')
    return d,fit,held,people,y


def capture(model,fit):
    result={}
    def train(self,X,y,sw,rows,epochs,lr,rng,gen,loss_fn,weighted,head_only=False,avg=None):
        self._prepare();result.update(X=X,y=y,rows=rows)
    model._train=types.MethodType(train,model);model.fit(fit)
    return result


def moments(x):
    frequency=torch.fft.rfftfreq(480,1/120);power=torch.fft.rfft(x,dim=-1).abs().square()
    blocks=[x.mean(-1)]
    for low,high in ((1,4),(4,8),(8,13),(13,20),(20,30),(30,55)):
        blocks.append(power[...,(frequency>=low)&(frequency<high)].mean(-1).clamp_min(1e-8).log())
    return torch.cat(blocks,1).numpy().astype(np.float64)


def embedding(model,rows):
    model.set_k(None);model.net.eval();values=[]
    def tap(module,args):values.append(args[0].detach().cpu().numpy().copy())
    handle=model.net.final_layer.register_forward_pre_hook(tap);before=state_sha(model.net)
    try:
        with torch.no_grad():
            for start in range(0,len(rows),128):model.net(torch.from_numpy(model._aligned(rows[start:start+128],True)))
    finally:handle.remove()
    if before!=state_sha(model.net):raise ValueError('embedding mutated native state')
    out=np.concatenate(values)
    if out.shape!=(len(rows),1152) or not np.isfinite(out).all():raise ValueError('native embedding layout')
    return out
