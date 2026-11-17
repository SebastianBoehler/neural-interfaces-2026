"""Source-person training and per-row out-of-person branch-reliability supervision."""
import hashlib
import time

import numpy as np
import torch
from torch import nn

from encoder import Encoder,fingerprint,moments
from views import Ranker,aggregate,calibrate,descriptors


def source_tensors(d,people,scale=None):
    rows=np.flatnonzero(np.isin(d['subject'],people))
    x=torch.from_numpy(d['X'][rows].astype(np.float32))
    if scale is None:
        scale=float(x.square().mean().sqrt())
    if not np.isfinite(scale) or scale<=0:
        raise ValueError('invalid source-only scalar')
    return rows,x/scale,torch.from_numpy(d['y'][rows]).long(),scale


def encoder_fit(x,y,epochs):
    start=time.monotonic(); net=Encoder('fb',0,'cpu')
    with torch.no_grad():
        m=moments(x);net.moment_mean.copy_(m.mean(0));net.moment_sd.copy_(m.std(0).clamp_min(1e-3))
    optimizer=torch.optim.AdamW(list(net.body.parameters())+list(net.task.parameters()),lr=.001,weight_decay=.001)
    generator=torch.Generator().manual_seed(320)
    weight=len(y)/(3*torch.bincount(y,minlength=3).float())
    if not torch.isfinite(weight).all():
        raise ValueError('source missing class')
    initial=fingerprint(net); logs=[]
    for ep in range(epochs):
        net.train(); losses=[]
        for rows in torch.randperm(len(x),generator=generator).split(64):
            optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(net(x[rows]),y[rows],weight=weight)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite source task loss')
            loss.backward();optimizer.step();losses.append(float(loss.detach()))
        logs.append(float(np.mean(losses)))
    net.eval()
    if fingerprint(net)==initial:
        raise ValueError('source encoder did not update')
    return net,dict(seconds=time.monotonic()-start,initial=initial,final=fingerprint(net),epochs=epochs,losses=logs)


def person_class_weights(people,y):
    weights=torch.zeros(len(y))
    for p in sorted(set(people)):
        for c in (0,1,2):
            rows=np.flatnonzero((people==p)&(y.numpy()==c))
            if len(rows)==0:
                raise ValueError('source person missing class')
            weights[rows]=1/len(rows)
    return weights/weights.sum()


def ranker_fit(logits,y,people,steps=400):
    features=descriptors(logits); model=Ranker(features.shape[-1])
    with torch.no_grad():
        model.mean.copy_(features.flatten(0,1).mean(0));model.sd.copy_(features.flatten(0,1).std(0).clamp_min(1e-3))
    losses=-logits.log_softmax(-1).gather(2,y[:,None,None].expand(-1,logits.shape[1],1)).squeeze(2)
    target=(losses[:,None,:]-losses[:,:,None]).sign().float()
    valid=(losses[:,None,:]-losses[:,:,None]).abs()>.01
    valid&=torch.triu(torch.ones(5,5,dtype=torch.bool),diagonal=1)[None]
    if not valid.any():
        raise ValueError('mask bank has no distinct source task losses')
    row_weights=person_class_weights(people,y)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    maximum=0.;logs=[]
    for step in range(steps):
        opt.zero_grad(set_to_none=True);score=model(features)
        delta=score[:,:,None]-score[:,None,:]
        loss=(nn.functional.softplus(-target*delta)*valid).sum((1,2))/valid.sum((1,2)).clamp_min(1)
        loss=(loss*row_weights).sum()
        if not torch.isfinite(loss):
            raise ValueError('nonfinite reliability ranking loss')
        loss.backward()
        gradient=max(float(p.grad.abs().max()) for p in model.parameters() if p.grad is not None)
        maximum=max(maximum,gradient);opt.step()
        if step in (0,steps-1):logs.append(float(loss.detach()))
    if maximum<=0:
        raise ValueError('ranker has no active gradients')
    model.eval()
    with torch.no_grad():
        scores=model(features);weights=scores.softmax(-1)
        static=-(losses*row_weights[:,None]).sum(0)
        probabilities={'H':logits[:,0].softmax(-1),'U':aggregate(logits,torch.zeros_like(scores)),
                       'S':aggregate(logits,static[None].expand_as(scores)),'R':aggregate(logits,scores)}
        temperatures={}
        for name,p in probabilities.items():
            values=[float((-(calibrate(p,t).log().gather(1,y[:,None]).squeeze())*row_weights).sum()) for t in (.5,1.,2.,4.)]
            temperatures[name]=(.5,1.,2.,4.)[int(np.argmin(values))]
    return model,static,temperatures,dict(max_gradient=maximum,losses=logs,source_weight_std=float(weights.std(0).max()),
              row_weight_sha256=hashlib.sha256(row_weights.numpy().tobytes()).hexdigest())
