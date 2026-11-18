"""Learn bounded per-channel adjustments after the original causal A6 normalization."""
import torch
from torch import nn


class Residual(nn.Module):
    def __init__(self,prefix):
        super().__init__()
        self.prefix=prefix
        self.read=nn.Sequential(nn.Linear(7,32),nn.SiLU(),nn.Linear(32,2))
        nn.init.zeros_(self.read[-1].weight)
        nn.init.zeros_(self.read[-1].bias)

    def forward(self,base):
        x=base.squeeze(2)
        absolute=x.abs()
        descriptors=torch.stack([x.mean(-1),absolute.mean(-1),
            *((absolute-k).relu().mean(-1) for k in (.5,1.,2.,4.))],-1)
        n=torch.arange(1,len(x)+1,device=x.device,dtype=x.dtype)[:,None,None]
        summary=descriptors.cumsum(0)/n if self.prefix else descriptors
        gate=n/(n+20)
        features=torch.cat([summary.asinh(),gate.expand(-1,x.shape[1],1)],-1)
        affine=.25*self.read(features).tanh()*gate
        gain=affine[:,:,0].exp()[:,:,None,None]
        shift=affine[:,:,1,None,None]
        return base*gain+shift


class NativeLayer(nn.Module):
    def __init__(self,original,residual):
        super().__init__()
        self.original=original
        self.residual=residual

    def forward(self,x):
        return self.residual(self.original(x))


class SourceModel(nn.Module):
    """Call certified fixed-48 TorchScript children with the adapter at the same layer."""
    def __init__(self,original,residual):
        super().__init__()
        self.original=original
        self.residual=residual

    def forward(self,x):
        net=self.original
        conv,norm,activation=net.spatial_conv.children()
        h=conv(net.spectral_filtering(x))
        h=activation(self.residual(norm(h)))
        # Certified bundle native.ts views this tensor as (n, channels, 4, 120).
        h=net.temporal_layer(h.reshape(len(h),h.shape[1],4,120))
        return net.final_layer(net.flatten_layer(h))
