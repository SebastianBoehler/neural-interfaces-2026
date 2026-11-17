"""Independent complex direct recurrence validates real scan outputs AND gradients."""
import copy
import torch
from torch.nn import functional as F


def direct_layer(layer,x):
    ar,ai,br,bi=layer.coefficients(x)
    a=torch.complex(ar,ai);b=torch.complex(br,bi)
    def recurse(transition,drive):
        h=torch.zeros_like(drive[:,0]);values=[]
        for t in range(drive.shape[1]):
            h=transition[:,t]*h+drive[:,t];values.append(h)
        return torch.stack(values,1)
    forward=recurse(a,b);reverse=recurse(a.flip(1),b.flip(1)).flip(1)
    z=.5*(forward+reverse)
    out=layer.mix(torch.cat([z.real,z.imag],-1))
    return x+F.gelu(out)*torch.sigmoid(layer.gate(layer.norm(x)))


def direct(model,x):
    z=model.stem(x-x.mean(1,keepdim=True)).transpose(1,2)
    for layer in model.layers:z=direct_layer(layer,z)
    body=torch.cat([z.mean(1),z.var(1,unbiased=False).clamp_min(1e-6).log()],1)
    moments=(model.moments(x)-model.moment_mean)/model.moment_sd
    return model.head(torch.cat([body,moments],1))


def compare(model,x,target,dtype):
    scan=copy.deepcopy(model).to(dtype);reference=copy.deepcopy(model).to(dtype)
    a=x.clone().to(dtype).requires_grad_(True);b=x.clone().to(dtype).requires_grad_(True)
    out=scan(a);expected=direct(reference,b)
    loss=F.cross_entropy(out,target);other=F.cross_entropy(expected,target)
    ga=torch.autograd.grad(loss,[a]+list(scan.parameters()),allow_unused=True)
    gb=torch.autograd.grad(other,[b]+list(reference.parameters()),allow_unused=True)
    names=['input']+[n for n,_ in scan.named_parameters()];errors={}
    for name,left,right in zip(names,ga,gb):
        if (left is None)!=(right is None):raise ValueError('oracle gradientpath mismatch:'+name)
        if left is not None:
            if not torch.isfinite(left).all() or not torch.isfinite(right).all():raise ValueError('nonfinite oraclegradient')
            errors[name]=float((left-right).abs().max())
    error=float((out-expected).abs().max());bound=1e-9 if dtype==torch.float64 else 2e-4
    if error>bound or max(errors.values())>bound:raise ValueError(f'{dtype}output/gradientoracle failed:{error},{max(errors.values())}')
    return dict(output_max_abs=error,gradient_max_abs=max(errors.values()),gradient_errors=errors,bound=bound)
