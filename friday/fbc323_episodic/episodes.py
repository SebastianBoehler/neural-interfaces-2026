"""Use two distinct class-balanced support/query sets inside each original source batch."""
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from anchors import row_sha


HERE = Path(__file__).resolve().parent
KINDS = ("hard", "within", "cross")
MODES = ("ordinary", "bounded")
ARMS = tuple(f"{k}_{m}" for k in KINDS for m in MODES)
ETA, WEIGHT, TOL = 2., .25, 2e-5
COUNT = 6


def organize(people, y, rows):
    if len(rows) not in (32,64):
        raise ValueError("native source chunks were not 32 windows")
    if len(rows)==32:
        return None
    left,right = rows[:32],rows[32:]
    distinct = [np.unique(people[part]) for part in (left,right)]
    if any(len(p)!=1 for p in distinct):
        raise ValueError("original chunk crossed source people")
    if distinct[0][0]==distinct[1][0]:
        return None
    role = []
    for offset,part in ((0,left),(32,right)):
        choices = []
        for label in (0,1,2):
            hit=np.flatnonzero(y[part]==label)
            _, index=np.unique(part[hit],return_index=True)
            found=hit[np.sort(index)]
            if len(found)<4:
                return None
            choices.append(offset+found[:4])
        role.append(np.stack(choices))
    slots = np.stack(role)
    support,query=(a.reshape(2,COUNT) for a in (slots[:,:,:2], slots[:,:,2:]))
    if (len(np.unique(rows[np.r_[support.ravel(),query.ravel()]]))!=4*COUNT or
            any(not np.array_equal(np.bincount(y[rows[p]],minlength=3),[2,2,2])
                for side in (support,query) for p in side)):
        raise ValueError("episode support/query roles overlap or have unequal class exposures")
    return support,query


def plan(source, people, labels, batches):
    plan, selected = [], []
    digest=hashlib.sha256()
    for step,rows in enumerate(r for e in batches for r in e):
        roles=organize(people,labels,rows)
        plan.append(roles)
        if roles is not None:
            selected.append(step)
            digest.update(np.asarray(step,dtype="<i8").tobytes())
            digest.update(np.asarray(source[rows[np.stack(roles)]],dtype="<i8").tobytes())
    if len(selected)<6 or len(selected)/len(plan)<.15:
        raise ValueError("not enough real balanced source episodes")
    return plan,selected[:6],dict(episode_sha=digest.hexdigest(), episode_steps=len(selected),
                                  support_exposure=2*COUNT*len(selected),query_exposure=2*COUNT*len(selected),
                                  diagnostic_steps=row_sha(selected[:6]), roles_disjoint=True,marginals_exact=True)


def unit(z):
    return F.normalize(z-z.mean(1,keepdim=True),dim=1)


def hypothetical(z, logits, labels, support, query, kind, rate=ETA):
    if kind=="cross":
        support=support.flip(0)
    elif kind!="within":
        raise ValueError("invalid temporary episode support")
    err=logits[support].softmax(-1)-F.one_hot(labels[support],3).to(z.dtype)
    delta=-(rate/COUNT)*(z[query] @ z[support].transpose(-1,-2) @ err)
    return logits[query]+delta,delta


def analytic(z, logits, labels, support, query, kind):
    a=support.flip(0) if kind=="cross" else support
    shifted,_=hypothetical(z,logits,labels,support,query,kind)
    zero,_=hypothetical(z,logits,labels,support,query,kind,0.)
    if not torch.equal(zero,logits[query]):
        raise ValueError("zero temporary step changed original logits")
    worst=0.
    for sv,qv,expected in zip(a,query,shifted,strict=True):
        weight=torch.zeros((3,z.shape[1]),device=z.device,requires_grad=True)
        inner=F.cross_entropy(logits[sv].detach()+z[sv].detach()@weight.T,labels[sv])
        measured,=torch.autograd.grad(inner,weight)
        value=logits[qv].detach()-ETA*z[qv].detach()@measured.T
        worst=max(worst,float((value-expected.detach()).abs().max()))
    if not 0<=worst<=TOL:
        raise ValueError("temporary step differs from independent source autograd")
    return worst
