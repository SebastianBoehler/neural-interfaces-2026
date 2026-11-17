"""Fixed body-feature masks; the287mean/power skip features remain untouched."""
import hashlib

import torch
from torch import nn

from encoder import fingerprint,moments


def bank(dimension):
    if dimension<4:
        raise ValueError('four nonempty masked feature partitions required')
    mask=torch.ones(5,dimension)
    for k in range(4):
        mask[k+1]*=4/3
        mask[k+1,k*dimension//4:(k+1)*dimension//4]=0
    return mask


def mask_sha(mask):
    return hashlib.sha256(mask.detach().numpy().tobytes()).hexdigest()


def branches(net,x,mask):
    net.eval(); before=fingerprint(net)
    with torch.no_grad():
        z=net.representation(x)
        skip=(moments(x)-net.moment_mean)/net.moment_sd
        features=z[:,None,:]*mask[None,:,:]
        repeated=skip[:,None,:].expand(-1,len(mask),-1)
        if not all(torch.equal(repeated[:,k],skip) for k in range(len(mask))):
            raise ValueError('mean/power skip changed across masks')
        logits=net.task(torch.cat([features,repeated],2))
    if fingerprint(net)!=before or any(m.training for m in net.modules() if isinstance(m,nn.modules.batchnorm._BatchNorm)):
        raise ValueError('multi-view inference changed population BN/model state')
    return logits.double()


def descriptors(logits):
    p=logits.softmax(-1); logp=logits.log_softmax(-1)
    identity=p[:,:1].expand_as(p)
    entropy=-(p*logp).sum(-1,keepdim=True)
    top=p.topk(2,-1).values
    branch=torch.eye(p.shape[1])[None].expand(len(p),-1,-1).to(p)
    return torch.cat([p,identity,(p-identity).abs(),entropy,top[...,:1],top[...,:1]-top[...,1:],branch],-1).float()


class Ranker(nn.Module):
    def __init__(self,features):
        super().__init__(); torch.manual_seed(320)
        self.net=nn.Sequential(nn.Linear(features,16),nn.Tanh(),nn.Linear(16,1))
        self.register_buffer('mean',torch.zeros(features)); self.register_buffer('sd',torch.ones(features))

    def forward(self,x):
        return self.net((x-self.mean)/self.sd).squeeze(-1)


def aggregate(logits,scores):
    if scores.shape!=logits.shape[:2]:
        raise ValueError('branch score shape mismatch')
    p=(logits.softmax(-1)*scores.double().softmax(-1)[...,None]).sum(1)
    if not torch.isfinite(p).all() or (p<=0).any() or not torch.allclose(p.sum(1),torch.ones(len(p),dtype=p.dtype)):
        raise ValueError('invalid weighted probabilities')
    return p


def calibrate(p,temperature):
    return (p.log()/temperature).softmax(-1)
