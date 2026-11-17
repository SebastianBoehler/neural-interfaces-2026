"""Explicit active paths, stateless batch/prefix guards, amplitude and stability."""
import copy
import numpy as np
import torch
from torch.nn import functional as F
from data331 import digest


def fingerprint(model):
    import hashlib
    h=hashlib.sha256()
    for n,v in sorted(model.state_dict().items()):h.update(n.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def gradients(model,x,y,update=False):
    x=x.detach().clone().requires_grad_(True)
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    model.zero_grad(set_to_none=True);F.cross_entropy(model(x),y).backward()
    if x.grad is None or not torch.isfinite(x.grad).all() or float(x.grad.abs().max())<=1e-12:raise ValueError("inputgradient inactive")
    result={};active=0;total=0
    for name,p in model.named_parameters():
        value=None if p.grad is None else float(p.grad.abs().max())
        mode=model.layers[0].mode
        if '.dt.' in name and mode!='selective':
            if value!=0:raise ValueError('static/local dtgradient notexactzero')
        elif name.endswith('.frequency') and mode=='local':
            if value is not None:raise ValueError('local frequency unexpectedpath')
        elif value is None or not np.isfinite(value) or value<=1e-12:
            raise ValueError('expected activegradient missing:'+name)
        result[name]=value;total+=p.numel()
        if value is not None and value>1e-12:active+=p.numel()
    if update:torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01).step()
    change={n:float((p.detach()-before[n]).abs().max()) for n,p in model.named_parameters()}
    if update:
        for name,value in result.items():
            if value is not None and value>1e-12 and change[name]<=0:raise ValueError('active optimizer no-op:'+name)
    return dict(input_gradient_max=float(x.grad.abs().max()),parameter_gradient_max=result,parameter_change_max=change,total_parameters=total,active_path_parameters=active)


def inference(model,x):
    model.eval();before=fingerprint(model)
    with torch.no_grad():
        p=model(x).softmax(1);single=torch.cat([model(v[None]).softmax(1) for v in x])
        error=float((p-single).abs().max())
        if error>2e-5:raise ValueError('B1/B4probability error')
        margin=p.topk(2,dim=1).values.diff(dim=1).abs().flatten()
        if torch.any((margin>1e-6)&(p.argmax(1)!=single.argmax(1))):raise ValueError('B1/B4nontiedargmax changed')
        prefix=model(x[:1]).softmax(1);model(x[1:]);repeat=model(x[:1]).softmax(1)
        copied=copy.deepcopy(model)(x[:1]).softmax(1)
        if not torch.equal(prefix,repeat) or not torch.equal(prefix,copied):raise ValueError('prefix/interleaving/deepcopy')
        amp=float((model(x[:1]*2).softmax(1)-prefix).abs().max())
        changed=x[:1].clone();changed[:,0,-1]+=1
        last=float((model(changed).softmax(1)-prefix).abs().max())
        if min(amp,last)<=1e-8:raise ValueError('amplitude/finalsample inactive')
        z=model.stem(x-x.mean(1,keepdim=True)).transpose(1,2);rho=[]
        for layer in model.layers:
            ar,ai,_,_=layer.coefficients(z);mod=(ar.square()+ai.square()).sqrt()
            if not torch.isfinite(mod).all() or torch.any(mod>=1) or torch.any(mod<=0):raise ValueError('noncontractive coefficient')
            rho.append([float(mod.min()),float(mod.max())]);z=layer(z)
    if fingerprint(model)!=before:raise ValueError('inference state mutated')
    return dict(B1_B4_max_abs=error,amplitude=amp,finalsample=last,transition_modulus=rho,state_sha256=before)
