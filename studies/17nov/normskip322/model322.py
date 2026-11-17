"""Fixed-head normalization × mean/power-skip factorial."""
import hashlib
import numpy as np
import torch
from encoder312 import Encoder, moments
from session_bn import SessionBN, swap_bn

SEED=0;M=32;BATCH=64;EPOCHS=40


class Factor(Encoder):
 def __init__(self,norm,skip,device):
  super().__init__('fb',SEED,device);self.skip=skip;self.norm=norm
  if norm=='session':
   count=swap_bn(self.body,False,20.)
   if count<1:raise ValueError('no BatchNorm was replaced')
   for m in self.body.modules():
    if isinstance(m,SessionBN):m.m=M;m.k=20.
  elif norm!='population':raise ValueError('bad norm')

 def forward(self,x):
  z=self.representation(x);q=(moments(x)-self.moment_mean)/self.moment_sd
  if not self.skip:q=torch.zeros_like(q)
  return self.task(torch.cat([z,q],1))


def batches(stream,epochs=EPOCHS):
 rng=np.random.default_rng(SEED);out=[]
 for epoch in range(epochs):
  chunks=[]
  for name in np.unique(stream):
   rows=np.flatnonzero(stream==name);rows=rows[rng.permutation(len(rows))]
   pad=(-len(rows))%M
   if pad:rows=np.r_[rows,rng.choice(rows,pad,replace=pad>len(rows))]
   chunks.extend(list(rows.reshape(-1,M)))
  chunks=[chunks[i] for i in rng.permutation(len(chunks))]
  out.append([np.concatenate(chunks[i:i+2]) for i in range(0,len(chunks),2)])
 return out


def param_hash(model):
 h=hashlib.sha256()
 for name,p in sorted(model.named_parameters()):h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
 return h.hexdigest()


def setup(d,fit,device,epochs=EPOCHS):
 x=torch.as_tensor(d['X'][fit],device=device).float(); y=torch.as_tensor(d['y'][fit],device=device).long()
 scale=float(x.square().mean().sqrt());x=x/scale
 if not np.isfinite(scale) or scale<=0:raise ValueError('invalid fit scale')
 with torch.no_grad():
  q=moments(x);mu=q.mean(0);sd=q.std(0).clamp_min(1e-3)
 counts=torch.bincount(y,minlength=3).float()
 if (counts==0).any():raise ValueError('fit class missing')
 return x,y,scale,mu,sd,len(y)/(3*counts),batches(d['stream'][fit],epochs)


def train(model,x,y,mu,sd,weights,all_batches):
 model.moment_mean.copy_(mu);model.moment_sd.copy_(sd)
 opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
 n=sum(len(a) for a in all_batches);sched=torch.optim.lr_scheduler.OneCycleLR(opt,.001,total_steps=n)
 torch.manual_seed(SEED);model.train();max_grad=0.;first=last=None
 for epoch in all_batches:
  for rows in epoch:
   opt.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(model(x[rows]),y[rows],weight=weights)
   if not torch.isfinite(loss):raise ValueError('nonfinite loss')
   loss.backward();grads=[p.grad for p in model.parameters() if p.grad is not None]
   if not grads or not all(torch.isfinite(g).all() for g in grads):raise ValueError('invalid gradients')
   max_grad=max(max_grad,max(float(g.abs().max()) for g in grads));opt.step();sched.step()
   last=float(loss.detach());first=last if first is None else first
 if max_grad<=0:raise ValueError('no training gradient')
 model.eval();return {'steps':n,'max_gradient':max_grad,'first_loss':first,'last_loss':last}
