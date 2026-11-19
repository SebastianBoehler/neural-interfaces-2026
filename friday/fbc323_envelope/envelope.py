"""Learn a parallel analytic or real-voltage power waveform readout."""
import hashlib
from pathlib import Path
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F


HERE=Path(__file__).resolve().parent
KINDS=("hard","analytic","real_power")
MODES=("ordinary","bounded")
ARMS=tuple(f"{k}_{m}" for k in KINDS for m in MODES)
BASE,EXTRA,FIXED,NATIVE,BRANCH=347903,47496,287,1152,256
MAX_RATIO,MAX_RESERVED=3.,8_000_000_000


def parameter_hash(module):
    digest=hashlib.sha256()
    for name,value in sorted(module.named_parameters()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Morphology(nn.Module):
    def __init__(self,kind,seed,device):
        super().__init__()
        self.kind=kind
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(34000+seed)
            self.spatial=nn.Parameter(torch.empty(9,8,41))
            nn.init.uniform_(self.spatial,-41**-.5,41**-.5)
            self.first=nn.Conv1d(72,64,5,padding=2,bias=False)
            self.first_norm=nn.GroupNorm(8,64)
            self.second=nn.Conv1d(64,64,5,padding=2,bias=False)
            self.second_norm=nn.GroupNorm(8,64)
        self.to(device)

    def spectrum(self,bands):
        if bands.shape[1:]!=(9,41,480):
            raise ValueError("unexpected original raw band geometry")
        wave=torch.einsum("fkc,nfct->nfkt",self.spatial,bands).flatten(1,2)
        s=torch.fft.rfft(wave.double(),dim=-1)
        return torch.cat((torch.zeros_like(s[...,:1]),s[...,1:-1],torch.zeros_like(s[...,-1:])),dim=-1)

    def powers(self,s):
        i=torch.fft.irfft(s,n=480,dim=-1).square()
        q=torch.fft.irfft(-1j*s,n=480,dim=-1).square()
        return .5*(i+q),i

    def tokens(self,power):
        return F.avg_pool1d(power/power.mean(-1,keepdim=True).clamp_min(1e-8),15,15)

    def features(self,tokens):
        first=F.gelu(self.first_norm(self.first(tokens.to(dtype=self.first.weight.dtype))))
        refined=first+F.gelu(self.second_norm(self.second(first)))
        return F.avg_pool1d(refined,8,8).flatten(1)

    def forward(self,bands):
        analytic,real=self.powers(self.spectrum(bands))
        use=analytic if self.kind=="analytic" else real
        return self.features(self.tokens(use))


class Head(nn.Module):
    def __init__(self,native,seed):
        super().__init__()
        self.native=native
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(34100+seed)
            self.emit=nn.Linear(BRANCH,3,bias=False)
        nn.init.zeros_(self.emit.weight)
        self.emit.to(native.weight.device)

    def forward(self,value):
        a,b,q=value.split((NATIVE,BRANCH,FIXED),dim=1)
        return self.native(torch.cat((a,q),dim=1))+self.emit(b)


def representation(model,x):
    body=model.body
    bands=body.spectral_filtering(x)
    grid=body.padding_layer(body.spatial_conv(bands)).view(len(x),288,4,120)
    power=body.final_layer(body.flatten_layer(body.temporal_layer(grid))).flatten(1)
    return torch.cat((power,model.morphology(bands)),dim=1)


def attach(model,kind,seed):
    if (model.skip or model.dim!=NATIVE or type(model.body.final_layer) is not nn.Identity or
            (model.task.in_features,model.task.out_features)!=(NATIVE+FIXED,3) or
            tuple(model.body.spectral_filtering.band_filters)!=tuple((n,n+4) for n in range(4,40,4))):
        raise ValueError("original fixed spectral bank or native hard head differs")
    if kind=="hard":
        return None
    if kind not in KINDS:
        raise ValueError("unexpected envelope mechanism")
    model.morphology=Morphology(kind,seed,model.task.weight.device)
    model.task=Head(model.task,seed)
    model.representation=MethodType(representation,model)
    return model.morphology
