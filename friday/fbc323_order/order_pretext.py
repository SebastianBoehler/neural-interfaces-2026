"""Match opposing temporal directions on gain-invariant interior patches."""
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


HERE=Path(__file__).resolve().parent
KINDS=("hard","ordered","sham")
MODES=("ordinary","bounded")
ARMS=tuple(f"{k}_{m}" for k in KINDS for m in MODES)
CHANNELS, LENGTH, SUBFRAME = 288,60,20
WEIGHT,TOL,PARAMETERS = .20,2e-5,28704


def tape(batches,held,seed):
    rng=np.random.default_rng(31031+100*int(held[1:])+seed)
    plan=[]
    signature=hashlib.sha256()
    total=flips=0
    for i,rows in enumerate(r for e in batches for r in e):
        slots=2+rng.permutation((np.arange(len(rows))+i)%3)
        signs=np.ones(len(rows),dtype=np.float32)
        for s in range(2,5):
            idx=np.flatnonzero(slots==s)
            flipped=rng.permutation(idx)[:len(idx)//2]
            signs[flipped]=-1
            if abs(signs[idx].sum())>1:
                raise ValueError("sham assignment became unbalanced within its actual time geometry")
        total+=len(rows)
        flips+=int((signs<0).sum())
        signature.update(slots.astype("<i8").tobytes())
        signature.update(signs.astype("<f4").tobytes())
        plan.append((slots,signs))
    return plan,dict(tape=signature.hexdigest(), pair_rows=total, orientation_examples=2*total,
                     sham_flips=flips, time_starts=[120,180,240], same_marginal=True)


def patches(value, slots):
    if value.shape[1:]!=(CHANNELS,4,120):
        raise ValueError("original native pre-pool geometry differs")
    tokens=value.reshape(len(value),CHANNELS,8,LENGTH).transpose(1,2)
    selector=torch.stack((slots,slots+1),dim=1)
    return tokens[torch.arange(len(value),device=value.device)[:,None],selector]


def describe(x):
    var=x.reshape(*x.shape[:-1],3,SUBFRAME).var(-1)
    relative=var/var.mean(-1,keepdim=True).clamp_min(torch.finfo(var.dtype).tiny)
    log=relative.clamp_min(1e-6).log()
    return F.normalize((log-log.mean(-1,keepdim=True)).flatten(-2),dim=-1)


class Arrow(nn.Module):
    def __init__(self,seed,device):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(31000+seed)
            self.project=nn.Linear(CHANNELS*3,32)
            self.a=nn.Linear(32,16,bias=False)
            self.b=nn.Linear(32,16,bias=False)
        self.to(device)

    def read(self,z):
        h=F.gelu(self.project(z))
        left,right=self.a(h).unbind(1)
        u,v=self.b(h).unbind(1)
        return (left*v-right*u).sum(1)/4.

    def forward(self,x):
        return self.read(describe(x))


@torch.no_grad()
def geometry(model, sample):
    band=model.body.spectral_filtering
    if band.method_iir or band.forward_filter or len(band.fir_list)!=9:
        raise ValueError("actual native filter phase or radius cannot qualify the temporal geometry")
    taps=[p.numel() for p in band.fir_list]
    radius=max(n//2 for n in taps)
    if radius>120 or any(n%2==0 for n in taps):
        raise ValueError(f"native FIR padding could leak into an auxiliary patch: {taps}")
    x=sample[:8]
    y=band(x)[...,120:360]
    extended=torch.cat((x[...,:120].flip(-1),x,x[...,-120:].flip(-1)),dim=-1)
    change=float((band(extended)[...,240:480]-y).abs().max())
    if not 0<=change<=TOL:
        raise ValueError(f"real FIR boundary extension affected auxiliary geometry: {change}")
    return dict(filter_lengths=taps, fir_radius=radius, padding_error=change)


@torch.no_grad()
def invariance(head,x):
    x=x[:8]
    z=describe(x)
    gain=torch.linspace(.8,1.2,CHANNELS,device=x.device)[None,None,:,None]
    offset=torch.linspace(-.05,.05,CHANNELS,device=x.device)[None,None,:,None]
    patch_scale=x.new_tensor([1.,1.2])[None,:,None,None]
    modified=describe(x*gain*patch_scale+offset/patch_scale)
    drift=float((modified-z).abs().max())
    score=head.read(z)
    rev=head.read(z.flip(1))
    tied=head.read(z[:, :1].expand_as(z))
    if not 0<=drift<=TOL or not torch.equal(score,-rev) or torch.count_nonzero(tied):
        raise ValueError(f"gain/offset or order reversal control failed: {drift}")
    return drift
