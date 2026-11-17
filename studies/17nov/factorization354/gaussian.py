"""Independent DEFT-equation implementation on strong FBC time/frequency tokens."""
import torch
from torch import nn
import torch.nn.functional as F


class Gaussian(nn.Module):
    def __init__(self,dynamic):
        super().__init__();self.dynamic=dynamic
        self.obs=nn.Sequential(nn.LayerNorm(32),nn.Linear(32,32))
        nn.init.zeros_(self.obs[-1].weight);nn.init.constant_(self.obs[-1].bias,4.)
        self.mean=nn.Parameter(torch.zeros(32));self.precision=nn.Parameter(torch.full((32,),-4.))
        if dynamic:
            self.bank=nn.Parameter(torch.eye(32)[None].repeat(16,1,1)+.001*torch.randn(16,32,32))
            self.diag=nn.Parameter(torch.zeros(16,32))
            self.generator=nn.Sequential(nn.LayerNorm(32),nn.Linear(32,64),nn.ReLU(),nn.Linear(64,16))
            nn.init.normal_(self.generator[-1].weight,std=.01);nn.init.zeros_(self.generator[-1].bias)

    def forward(self,z,other=None,gate=None):
        b,w,d=z.shape;mean=self.mean[None].expand(b,-1);lp=self.precision[None].expand(b,-1)
        observation=self.obs(z);means=[];precisions=[];variations=[]
        for t in range(w):
            if self.dynamic:
                alpha=self.generator(mean).softmax(-1)
                transition=torch.einsum('bn,nij->bij',alpha,self.bank)
                mean=torch.einsum('bij,bj->bi',transition,mean)
                scale=alpha@(F.softplus(self.diag)+.001)
                lp=lp+scale.clamp_min(1e-6).log();variations.append(alpha.std(0,unbiased=False).max())
            value=z[:,t];op=observation[:,t]
            if other is not None:
                opposite,opposite_lp=other;effective=gate.sigmoid();value=value-effective*opposite[:,t]
                rp=opposite_lp[:,t]+effective.log()
                op=op+rp-torch.logaddexp(op,rp)
            total=torch.logaddexp(lp,op)
            mean=(lp-total).exp()*mean+(op-total).exp()*value;lp=total
            means.append(mean);precisions.append(lp)
        variation=torch.stack(variations).max() if variations else z.new_zeros(())
        return torch.stack(means,1),torch.stack(precisions,1),variation


class Recursive(nn.Module):
    def __init__(self):
        super().__init__();self.layers=nn.ModuleList([Gaussian(False),Gaussian(True),Gaussian(False),Gaussian(True)])
        self.removal=nn.Parameter(torch.full((3,),-2.))

    def forward(self,z):
        result=None;variations=[]
        for index,layer in enumerate(self.layers):
            mean,lp,variation=layer(z,None if result is None else result,
                                    None if index==0 else self.removal[index-1])
            result=(mean,lp);variations.append(variation)
            if index==2:style=mean
        return mean,style,torch.stack(variations).max()
