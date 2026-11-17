"""Observe all original65 and auxiliary neural parameters/non-stream buffers and reset state."""
import hashlib

import numpy as np
import torch


def neural_state(model):
    digest=hashlib.sha256();state=[];ids=[]
    def tensor(name,value):
        a=value.detach().cpu().numpy();digest.update(name.encode());digest.update(str(a.dtype).encode());digest.update(str(a.shape).encode());digest.update(a.tobytes())
    def module(prefix,value):
        for name,p in value.named_parameters():tensor(prefix+'/'+name,p)
        for name,b in value.named_buffers():
            if name.rsplit('.',1)[-1] in ('s1','s2','n'):state.append((prefix+'/'+name,float(b.abs().sum())))
            else:tensor(prefix+'/'+name,b)
    for _,members in model.nets.groups:
        for i,net in members:
            ids.append(i)
            if hasattr(net,'m'):module('original'+str(i),net.m)
            else:
                for name,value in net.t.items():tensor('original'+str(i)+'/'+name,value)
    expected=np.flatnonzero(model.w[1:-1]!=0).tolist()
    if len(model.w[1:-1])!=65 or sorted(ids)!=expected:raise ValueError('original declared/nonzero-weight runtime coverage changed')
    for i,m in enumerate(model.eye.nets):module('eye'+str(i),m)
    for i,m in enumerate(model.rest.nets):module('rest'+str(i),m)
    if model.js_bank is not None:
        for _,members in model.js_bank.groups:
            for i,net in members:module('aux'+str(i),net.m)
    return dict(immutable_sha256=digest.hexdigest(),original_declared_members=65,original_active_members=len(ids),original_inactive_ids=np.flatnonzero(model.w[1:-1]==0).tolist(),stream_buffer_count=len(state),
                all_stream_buffers_zero=all(v==0 for _,v in state),nonzero_stream_buffers=sum(v!=0 for _,v in state),
                core1_n=model.core1.n,core2_n=model.core2.n,original_input_n=[stream.n for stream,_ in model.nets.groups],
                auxiliary_input_n=model.js_bank.groups[0][0].n if model.js_bank is not None else 0,eye_n=model.eye.n,
                rest_input_n=model.rest.stream.n,ft_labels=len(model.ft.lab),cp_t=model.cp.t)


def rng():
    return dict(cpu=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                cuda=hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest())
