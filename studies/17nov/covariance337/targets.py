"""Full51-block source-fixed teacher; equal diagonal/offdiagonal/slow loss groups."""
import numpy as np
import torch
from torch import nn

D=44239
u=np.triu_indices(41)
DIAGONAL=np.concatenate([np.flatnonzero(u[0]==u[1])+861*b for b in range(51)])
OFFDIAGONAL=np.concatenate([np.flatnonzero(u[0]!=u[1])+861*b for b in range(51)])
SLOW=np.arange(43911,D)
GROUPS=(DIAGONAL,OFFDIAGONAL,SLOW)
if tuple(map(len,GROUPS))!=(2091,41820,328) or not np.array_equal(np.sort(np.concatenate(GROUPS)),np.arange(D)):raise ValueError('full target group coverage')


def scaler(x):
    if x.ndim!=2 or x.shape[1]!=D or not np.isfinite(x).all():raise ValueError('teacher schema')
    mean=x.mean(0);std=x.std(0)
    if np.any(std<=1e-12):raise ValueError('constant/invalid teacher coordinate')
    z=(x-mean)/std
    if np.max(np.abs(z.mean(0)))>1e-9 or np.max(np.abs(z.std(0)-1))>1e-9:raise ValueError('teacher source standardization')
    return mean,std


class TeacherHead(nn.Module):
    """Train-only direct1152→44239; preserves caller RNG and native classifier."""
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(337);self.linear=nn.Linear(1152,D)
        for i,group in enumerate(GROUPS):self.register_buffer(f'group{i}',torch.tensor(group))

    def forward(self,x,target):
        if x.shape!=(len(target),1152) or target.shape!=(len(x),D):raise ValueError('native feature/teacher dimensions')
        error=(self.linear(x)-target).square()
        losses=torch.stack([error[:,getattr(self,f'group{i}')].mean() for i in range(3)])
        return losses.mean(),losses
