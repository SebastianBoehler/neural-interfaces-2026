"""Independent mean/variance choices; donor moments follow their own upstream path."""
import copy
import torch
from session_bn import SessionBN
from modes import state_equal,views


class MomentBN(SessionBN):
    def forward(self,x):
        if self.training:raise ValueError('332inferenceonly')
        mode=self.mode332
        if mode not in ('M','V','D','B'):raise ValueError('unregisteredmomentview')
        donor=x
        if mode=='D':
            if len(x)%2:raise ValueError('target/sourcepairedbatchrequired')
            donor=x[len(x)//2:]
        red=tuple(range(2,x.dim()));r1=donor.mean(red) if red else donor
        r2=donor.square().mean(red) if red else donor.square()
        n=torch.arange(1,len(donor)+1,device=x.device,dtype=x.dtype)[:,None]
        mean=(20*self.running_mean+r1.cumsum(0))/(20+n)
        second=(20*(self.running_var+self.running_mean.square())+r2.cumsum(0))/(20+n)
        variance=(second-mean.square()).clamp_min(0)
        normmean=mean if mode in ('M','B','D') else self.running_mean.expand_as(mean)
        normvar=self.running_var.expand_as(variance) if mode=='M' else variance
        if mode=='D':normmean=torch.cat([normmean,normmean]);normvar=torch.cat([normvar,normvar])
        shape=(len(x),-1)+(1,)*len(red)
        out=(x-normmean.reshape(shape))/torch.sqrt(normvar.reshape(shape)+self.eps)
        return self._affine(out,(1,-1)+(1,)*len(red))


def prepare(model,probe):
    fixed,both=views(model,probe);result={'F':fixed,'B':both};counts={}
    for mode in ('M','V','D'):
        child=copy.deepcopy(both);count=0
        for layer in child.body.modules():
            if isinstance(layer,SessionBN):
                layer.__class__=MomentBN;layer.mode332=mode;count+=1
        if count<1:raise ValueError('noactualBNintervention')
        state_equal(both,child);result[mode]=child;counts[mode]=count
    return result,counts


def direct(layer,x,mode):
    """Slow independently accumulated per-prefix moment oracle on realactivations."""
    donor=x[len(x)//2:] if mode=='D' else x
    red=tuple(range(2,x.dim()));total=20*layer.running_mean;second=20*(layer.running_var+layer.running_mean.square())
    values=[]
    for i,z in enumerate(donor):
        axes=tuple(a-1 for a in red)
        first=z.mean(axes) if axes else z;square=z.square().mean(axes) if axes else z.square()
        total=total+first;second=second+square;mean=total/(21+i);var=(second/(21+i)-mean.square()).clamp_min(0)
        mu=mean if mode in ('M','B','D') else layer.running_mean
        vv=layer.running_var if mode=='M' else var
        values.append((mu,vv))
    output=[]
    for i,z in enumerate(x):
        mu,vv=values[i%len(values)];shape=(-1,)+(1,)*len(red)
        v=(z-mu.reshape(shape))/torch.sqrt(vv.reshape(shape)+layer.eps)
        output.append(v if layer.weight is None else v*layer.weight.reshape(shape)+layer.bias.reshape(shape))
    return torch.stack(output)
