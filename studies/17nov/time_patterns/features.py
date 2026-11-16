"""Stateless time-local spectrum and fixed temporal-pattern features (§314)."""
import numpy as np
import torch
import torch.nn.functional as F

BANDS=((1,3),(2,4),(3,6),(4,8),(6,10),(8,12),(10,14),(12,16),(14,18),
       (16,20),(18,24),(20,28),(24,32),(28,36),(32,40),(36,45),(40,58))
TF_DIM=41*4+41*4*17
DYNAMIC_DIM=16*12*4*4*2


def fixed_bank(seed):
    if seed not in (0,1): raise ValueError('unregistered bank seed')
    rng=np.random.default_rng(31400+seed)
    pairs=np.array([rng.choice(41,2,replace=False) for _ in range(16)],dtype=np.int64)
    kernels=rng.normal(size=(16,12,9));kernels-=kernels.mean(axis=-1,keepdims=True)
    kernels/=np.linalg.norm(kernels,axis=-1,keepdims=True)
    if any(np.array_equal(k,k[::-1]) for k in kernels.reshape(-1,9)): raise ValueError('symmetric temporal filter')
    return pairs,torch.as_tensor(kernels.reshape(192,1,9),dtype=torch.float32)


def extract(raw, rms, seed):
    if raw.ndim!=3 or raw.shape[1:]!=(41,480) or rms.shape!=(41,): raise ValueError('expected N×41×480 and 41 scales')
    if not np.isfinite(raw).all() or not np.isfinite(rms).all() or np.any(rms<=0): raise ValueError('invalid raw features/scaler')
    x=torch.as_tensor(np.ascontiguousarray(raw),dtype=torch.float32)/torch.as_tensor(rms,dtype=torch.float32)[None,:,None]
    bins=x.reshape(len(x),41,4,120); mean=bins.mean(dim=-1)
    fft=torch.fft.rfft(bins-mean.unsqueeze(-1),dim=-1); power=fft.abs().square()/120**2
    pieces=[mean.reshape(len(x),-1)]
    for lo,hi in BANDS: pieces.append(power[...,lo:hi].mean(-1).clamp_min(1e-12).log().reshape(len(x),-1))
    tf=torch.cat(pieces,dim=1)
    pairs,kernels=fixed_bank(seed);detrended=x-x.mean(-1,keepdim=True)
    projected=detrended[:,pairs[:,0]]-detrended[:,pairs[:,1]]
    dynamic=[]
    for dilation in (1,4,16,32):
        response=F.conv1d(projected,kernels,groups=16,dilation=dilation,padding=4*dilation).reshape(len(x),16,12,4,120)
        dynamic.append(torch.stack(((response>0).float().mean(-1),response.amax(-1)),dim=-1))
    dynamic=torch.stack(dynamic,dim=3).reshape(len(x),-1)
    if tf.shape!=(len(x),TF_DIM) or dynamic.shape!=(len(x),DYNAMIC_DIM): raise ValueError('feature dimension changed')
    if not torch.isfinite(tf).all() or not torch.isfinite(dynamic).all(): raise ValueError('nonfinite transform')
    return tf.numpy(),dynamic.numpy()
