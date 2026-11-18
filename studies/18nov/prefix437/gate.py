"""Frozen population and causal A6 experts with a strictly past reliability gate."""
import torch
from torch import nn
from torch.nn import functional as F


def endpoints(net,x,eps):
    conv,norm,activation=net.spatial_conv.children()
    h=conv(net.spectral_filtering(x))
    pop=F.batch_norm(h,norm.running_mean,norm.running_var,norm.weight,norm.bias,False,0.,eps)
    def head(raw):
        v=activation(raw)
        v=net.temporal_layer(v.reshape(len(v),v.shape[1],4,120))
        return net.final_layer(net.flatten_layer(v))
    return head(norm(h)),head(pop)


def features(k,p):
    a,b=k.softmax(1),p.softmax(1)
    desc=torch.stack([(a-b).abs().sum(1),a.amax(1)-b.amax(1)],1)
    n=torch.arange(len(a),device=a.device,dtype=a.dtype)[:,None]
    past=torch.cat([torch.zeros_like(desc[:1]),desc.cumsum(0)[:-1]])/n.clamp_min(1)
    return torch.cat([n/(n+20),past],1)


class Gate(nn.Module):
    def __init__(self,conditional):
        super().__init__()
        self.bias=nn.Parameter(torch.tensor(2.1972246))
        self.slopes=nn.Parameter(torch.zeros(3)) if conditional else None

    def forward(self,features):
        value=self.bias if self.slopes is None else self.bias+features@self.slopes
        return value.sigmoid().reshape(-1,1)


def mix(k,p,g):
    return g*k+(1-g)*p
