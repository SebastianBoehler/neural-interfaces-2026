"""Portable training kernel for an initialized A6 waveform model (Python or scripted)."""
import hashlib
import time
import numpy as np
import torch
from torch.nn import functional as F

GROUPS={'body':'spatial_conv.0.','norm':'spatial_conv.1.','head':'final_layer.'}


def sha_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def schedule(streams,seed,steps=120):
    rng=np.random.default_rng(seed)
    ss=[np.flatnonzero(streams==s) for s in sorted(set(streams))]
    return np.stack([rng.choice(ss[rng.integers(len(ss))],48,replace=False) for _ in range(steps)])


def configure(net,arm):
    assert arm in ('F','J','K')
    groups={g:[] for g in GROUPS}
    for name,p in net.named_parameters():
        matches=[key for key,token in GROUPS.items() if name.startswith(token)]
        assert len(matches)<=1
        enabled=bool(matches and (arm!='F' or matches[0]=='norm'))
        p.requires_grad_(enabled)
        if enabled:
            groups[matches[0]].append(p)
    required={'norm'} if arm=='F' else set(GROUPS)
    assert {k for k,v in groups.items() if v}==required
    return {g:v for g,v in groups.items() if v}


def group_digest(parameters):
    h=hashlib.sha256()
    for p in parameters:
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train(net,x,y,weights,streams,arm,seed,steps):
    start=time.monotonic()
    assert x.shape==(len(y),41,480) and len(streams)==len(y)==len(weights)
    net.eval()
    params=configure(net,arm)
    before={g:group_digest(ps) for g,ps in params.items()}
    frozen=[p for p in net.parameters() if not p.requires_grad]
    frozen_sha=group_digest(frozen)
    opt=torch.optim.AdamW([dict(params=ps,lr=.0003 if g=='norm' else .00003) for g,ps in params.items()],
                           weight_decay=.001)
    batches=schedule(streams,seed,steps)
    gradients={}
    losses=[]
    for b in batches:
        opt.zero_grad()
        pred=net(x[b])
        loss=(F.cross_entropy(pred,y[b],reduction='none')*weights[b]).mean()
        assert torch.isfinite(loss)
        loss.backward()
        if not gradients:
            for g,ps in params.items():
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in ps)
                gradients[g]=max(p.grad.abs().max().item() for p in ps)
                assert gradients[g]>0
        opt.step()
        losses.append(loss.item())
    after={g:group_digest(ps) for g,ps in params.items()}
    assert all(before[g]!=after[g] for g in params) and group_digest(frozen)==frozen_sha
    return dict(groups=list(params),before=before,after=after,first_grad=gradients,
                frozen_sha=frozen_sha,batches=sha_array(batches),seed=seed,steps=steps,
                first8_ce=float(np.mean(losses[:8])),last8_ce=float(np.mean(losses[-8:])),
                seconds=time.monotonic()-start)
