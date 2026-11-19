"""Keep the native S1C path and add signed normalized spatial relations."""
import hashlib
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


HERE=Path(__file__).resolve().parent
KINDS=("hard","aligned","permuted")
MODES=("ordinary","bounded")
ARMS=tuple(f"{k}_{m}" for k in KINDS for m in MODES)
BASE, EXTRAS, NATIVE, WIDTH, FIXED = 347903,1704,1152,240,287
MAX_RATIO, MAX_RESERVED = 2.5,8_000_000_000
SAMPLES=120
BANDS=((4.,8.),(8.,12.),(12.,16.))


def order(kind):
    normal=np.arange(SAMPLES)
    if kind=="aligned":
        return normal
    if kind=="permuted":
        rng=np.random.default_rng(33033)
        while True:
            candidate=rng.permutation(normal)
            if not (candidate==normal).any():
                return candidate
    raise ValueError("unsupported relation alignment")


def parameter_hash(module):
    digest=hashlib.sha256()
    for name,value in sorted(module.named_parameters()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def center(x):
    return x-x.mean(dim=-1,keepdim=True)


def cosine(a,b):
    return F.normalize(center(a),dim=-1,eps=1e-8)@F.normalize(center(b),dim=-1,eps=1e-8).transpose(-1,-2)


class Relations(nn.Module):
    def __init__(self,kind,seed,device):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(33000+seed)
            self.left=nn.Parameter(torch.empty(3,4,41))
            self.right=nn.Parameter(torch.empty(3,4,41))
            for p in (self.left,self.right):
                nn.init.uniform_(p,-41**-.5,41**-.5)
        self.register_buffer("permutation",torch.as_tensor(order(kind)))
        self.to(device)

    def project(self,spectra):
        bands=spectra[:,:len(BANDS)]
        return tuple(torch.einsum("fkc,nfct->nfkt",p,bands) for p in (self.left,self.right))

    def shuffled(self,right,permutation):
        return right.reshape(len(right),3,4,4,SAMPLES).index_select(-1,permutation).reshape_as(right)

    def describe(self,left,right):
        wide=cosine(left,right).flatten(1)
        blocks=[p.reshape(len(p),3,4,4,SAMPLES).transpose(2,3) for p in (left,right)]
        narrow=cosine(*blocks).flatten(1)
        return torch.cat((wide,narrow),dim=1)

    def forward(self,spectra):
        left,right=self.project(spectra)
        return self.describe(left,self.shuffled(right,self.permutation))


class Head(nn.Module):
    def __init__(self,native,seed):
        super().__init__()
        self.native=native
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(33100+seed)
            self.relation=nn.Linear(WIDTH,3,bias=False)
        nn.init.zeros_(self.relation.weight)
        self.relation.to(native.weight.device)

    def forward(self,complete):
        backbone,relations,fixed=complete.split((NATIVE,WIDTH,FIXED),dim=1)
        return self.native(torch.cat((backbone,fixed),dim=1))+self.relation(relations)


def represent(model,x):
    body=model.body
    bands=body.spectral_filtering(x)
    activation=body.padding_layer(body.spatial_conv(bands))
    blocks=activation.view(len(x),288,4,SAMPLES)
    native=body.final_layer(body.flatten_layer(body.temporal_layer(blocks))).flatten(1)
    return torch.cat((native,model.relations(bands)),dim=1)


def attach(model,kind,seed):
    b=model.body
    if (model.skip or type(b.final_layer) is not nn.Identity or model.dim!=NATIVE or
            (b.n_bands,b.n_filters_spat,b.n_times_padded,b.stride_factor)!=(9,32,480,4) or
            tuple(b.spectral_filtering.band_filters[:len(BANDS)])!=BANDS or
            (model.task.in_features,model.task.out_features)!=(NATIVE+FIXED,3)):
        raise ValueError("native bands, pool or hard S1C head differs")
    if kind=="hard":
        return None
    model.relations=Relations(kind,seed,model.task.weight.device)
    model.task=Head(model.task,seed)
    model.representation=MethodType(represent,model)
    return model.relations
