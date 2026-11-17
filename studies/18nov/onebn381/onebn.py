"""One learned affine replaces the actual A6 normalization layer."""
from pathlib import Path
import copy
import importlib.util
import torch
from torch import nn

OWNER=Path('/mnt/lustre/work/gehler/gfh527/ni26_joint373_v2_20261117')
spec=importlib.util.spec_from_file_location('original373',OWNER/'pilot.py')
joint=importlib.util.module_from_spec(spec)
spec.loader.exec_module(joint)
joint.check_manifest(OWNER,'5269d87eb3c10b5d6ead65745fd126915b101ba7b1aebf524ed9356a41513dab')


class LearnedBN(nn.Module):
    def __init__(self,original):
        super().__init__()
        self.register_buffer('mean',original.running_mean.clone())
        self.register_buffer('std',(original.running_var+original.eps).sqrt())
        self.register_buffer('weight',original.weight.detach().clone())
        self.register_buffer('bias',original.bias.detach().clone())
        self.eps=original.eps
        self.register_buffer('hinges',torch.tensor([.5,1.,2.,4.]))
        self.threshold=nn.Parameter(torch.zeros(4))
        self.read=nn.Sequential(nn.Linear(7,32),nn.SiLU(),nn.Linear(32,2))
        nn.init.zeros_(self.read[-1].weight)
        nn.init.zeros_(self.read[-1].bias)
        self.mode='real'
        self.donor=None

    def standardized(self,x):
        return (x-self.mean[None,:,None,None])/self.std[None,:,None,None]

    def encode(self,z):
        value=z.squeeze(2)
        absolute=value.abs()
        cutoff=self.hinges*(.25*self.threshold.tanh()).exp()
        return torch.stack([value.mean(-1),absolute.mean(-1),
                            *((absolute-v).relu().mean(-1) for v in cutoff)],-1)

    def affine(self,items,mode,donor=None):
        n=len(items)
        count=torch.arange(1,n+1,device=items.device,dtype=items.dtype)[:,None,None]
        if mode=='blank':
            count=torch.ones_like(count)
            pooled=items
        elif mode=='real':
            pooled=items.cumsum(0)/count
        elif mode=='donor':
            assert donor is not None
            sampled=donor[torch.arange(n,device=items.device)%len(donor)]
            earlier=torch.cat((torch.zeros_like(items[:1]),sampled[:-1].cumsum(0)))
            pooled=(items+earlier)/count
        else:
            raise ValueError(mode)
        mix=count/(count+20)
        inp=torch.cat((pooled.asinh(),mix.expand(-1,items.shape[1],1)),-1)
        raw=self.read(inp)
        gain_log=3.5*(raw[:,:,0]/3.5).tanh()*mix[:,:,0]
        shift=4*(raw[:,:,1]/4).tanh()*mix[:,:,0]
        return gain_log,shift

    def transform(self,z,log_gain,shift):
        scale=self.weight*log_gain.exp()
        offset=self.bias+self.weight*shift
        return z*scale[:,:,None,None]+offset[:,:,None,None]

    def forward(self,x):
        z=self.standardized(x)
        return self.transform(z,*self.affine(self.encode(z),self.mode,self.donor))


class Tail(nn.Module):
    def __init__(self,net):
        super().__init__()
        self.activation=copy.deepcopy(net.spatial_conv[2])
        self.padding=copy.deepcopy(net.padding_layer)
        self.pool=copy.deepcopy(net.temporal_layer)
        self.flatten=copy.deepcopy(net.flatten_layer)
        self.head=copy.deepcopy(net.final_layer)
        self.stride=net.stride_factor
        self.n_times=net.n_times_padded
        self.requires_grad_(False)

    def forward(self,x):
        x=self.padding(self.activation(x))
        return self.head(self.flatten(self.pool(x.reshape(len(x),x.shape[1],self.stride,self.n_times//self.stride))))


def preactivation(net,x):
    return net.spatial_conv[0](net.spectral_filtering(x))


def source():
    import numpy as np
    torch.set_num_threads(2)
    d,_,_,people=joint.load()
    rows=np.flatnonzero(np.isin(people,joint.FIT_PEOPLE))
    assert 'y' not in d and set(people[rows])==set(joint.FIT_PEOPLE)
    runtime=joint.restore(d,rows,joint.verify_body())
    assert len(runtime.sbns())==1 and runtime.net.spatial_conv[1] is runtime.sbns()[0]
    assert len(runtime.net.spatial_conv)==3 and torch.get_num_threads()==2
    return d,people,rows,runtime
