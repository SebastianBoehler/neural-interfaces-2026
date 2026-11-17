"""Five fixed controls; FBC's native9frequency×4time×32spatial token grid."""
import torch
from torch import nn
import torch.nn.functional as F
from gaussian import Recursive


def mlp(hidden):return nn.Sequential(nn.LayerNorm(32),nn.Linear(32,hidden),nn.GELU(),nn.Linear(hidden,32))


class Adapter(nn.Module):
    def __init__(self,arm):
        super().__init__();self.arm=arm;self.cache=None
        if arm=='M':self.content=mlp(722)
        elif arm=='L':self.content=mlp(344);self.style=mlp(344)
        elif arm=='G':self.recursive=Recursive()
        else:raise ValueError('registered adapter arm required')
        if arm in ('L','G'):self.film=nn.Linear(32,64)

    def forward(self,tokens):
        mixed=tokens.mean(1)
        if self.arm=='M':content=mixed+self.content(mixed);style=None;variation=mixed.new_zeros(())
        elif self.arm=='L':content=mixed+self.content(mixed);style=self.style(mixed);variation=mixed.new_zeros(())
        else:content,style,variation=self.recursive(mixed)
        if self.training:self.cache=(mixed,content,style,variation)
        elif self.cache is not None:raise ValueError('training cache crossed into inference')
        return tokens-mixed[:,None]+content[:,None]

    def losses(self,people):
        if self.cache is None:raise ValueError('missing actual adapter tokens')
        mixed,content,style,variation=self.cache
        if self.arm=='M':return {},variation
        vc=(content-content.mean(1,keepdim=True)).square().mean((1,2))
        vs=(style-style.mean(1,keepdim=True)).square().mean((1,2))
        stable=style.mean(1);embedding=F.normalize(stable,dim=1)
        similarity=embedding@embedding.T/.07;eye=torch.eye(len(people),device=people.device,dtype=torch.bool)
        positive=(people[:,None]==people[None])&~eye;negative=people[:,None]!=people[None]
        if not positive.any(1).all() or not negative.any(1).all():raise ValueError('subject contrast lacks positives/negatives')
        logprob=similarity-similarity.masked_fill(eye,-torch.inf).logsumexp(1,keepdim=True)
        contrast=-(logprob.masked_fill(~positive,0).sum(1)/positive.sum(1)).mean()
        scale,bias=self.film(stable).chunk(2,-1);reconstruction=scale[:,None]*content+bias[:,None]
        losses=dict(order=(.01+vs-vc).clamp_min(0).mean(),style=vs.mean(),
                    contrast=contrast,reconstruction=(reconstruction-mixed).square().mean())
        return losses,variation


class TokenLayer(nn.Module):
    def __init__(self,inner,arm):
        super().__init__();self.inner=inner;self.adapter=Adapter(arm)

    def forward(self,x):
        value=self.inner(x)
        if value.shape[1:]!=(288,4,1):raise ValueError('actual strong FBC temporal interface changed')
        tokens=value[...,0].reshape(len(value),9,32,4).permute(0,1,3,2)
        content=self.adapter(tokens)
        return content.permute(0,1,3,2).reshape(len(value),288,4,1)
