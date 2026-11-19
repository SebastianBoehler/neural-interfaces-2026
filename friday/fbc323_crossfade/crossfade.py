"""Apply matched genuine and same-trial smooth suffix substitutions in source only."""
from pathlib import Path
from types import MethodType

import numpy as np
import torch


HERE=Path(__file__).resolve().parent
KINDS=("hard","cross_trial","same_trial")
MODES=("ordinary","bounded")
ARMS=tuple(f"{k}_{m}" for k in KINDS for m in MODES)
PARAMETERS=347903
LOW,HIGH,SHIFT=120,360,60
TOL=2e-6


def ramp(device):
    fade=torch.zeros(480,dtype=torch.float64,device=device)
    edge=(1-torch.linspace(0,torch.pi,31,dtype=torch.float64,device=device).cos())*.5
    fade[120:151]=edge
    fade[151:330]=1
    fade[330:361]=edge.flip(0)
    if (fade[:121].count_nonzero() or fade[360:].count_nonzero() or not torch.all(fade[150:331]==1) or
            not 0<float(fade.diff().abs().max())<=.053):
        raise ValueError("wrong smooth common crossfade support")
    return fade[LOW:HIGH+1]


@torch.no_grad()
def transforms(receivers,mates,shift,mask,mass=.5):
    clean=receivers[...,LOW:HIGH+1].double()
    pos=torch.arange(LOW,HIGH+1,device=receivers.device)[None,:]+shift[:,None]
    sham=torch.gather(receivers,-1,pos[:,None,:].expand(-1,41,-1)).double()
    delta=torch.stack((mates[...,LOW:HIGH+1].double()-clean,sham-clean))
    native=clean.flatten(1).norm(dim=1)
    want=delta*mask
    sizes=want.flatten(2).norm(dim=2)
    target=torch.minimum(sizes.amin(0),mass*native)
    if not (native>0).all() or not (sizes>0).all() or mass and not (target>0).all():
        raise ValueError("inactive real physical source displacement")
    gain=target[None,:]/sizes
    out=receivers.repeat(2,1,1,1)
    out[...,LOW+1:HIGH]=(clean[None,...]+gain[:,:,None,None]*want)[...,1:-1].to(receivers.dtype)
    actual=((out.double()-receivers.double()).flatten(2).norm(dim=2)/native)
    norm_error=(actual[0]-actual[1]).abs().max()
    seam=mask.diff()*delta[...,1:]*gain[:,:,None,None]
    before=clean.diff().flatten(1).norm(dim=1)
    seam_ratio=seam.flatten(2).norm(dim=2)/before.clamp_min(1e-8)
    movement=(out[0]-out[1]).abs().max()
    if (not torch.isfinite(out).all() or not torch.isfinite(seam_ratio).all() or norm_error>TOL or
            mass and ((actual<=0).any() or not movement>1e-4) or
            torch.any(actual>mass+TOL) or
            not torch.equal(out[...,:LOW+1],receivers[None,...,:LOW+1].expand_as(out[...,:LOW+1])) or
            not torch.equal(out[...,HIGH:],receivers[None,...,HIGH:].expand_as(out[...,HIGH:]))):
        raise ValueError("crossfade support, exact raw RMS dose or treatment geometry failed")
    if not mass and not torch.equal(out,receivers[None].expand_as(out)):
        raise ValueError("actual zero-dose path changed original inputs")
    return out,torch.stack((norm_error,actual[0].mean(),actual[1].mean(),seam_ratio[0].mean(),
                            seam_ratio[1].mean(),movement.double(),gain[0].mean(),gain[1].mean()))


class Feed:
    def __init__(self,original,flat,tape,bounds,kind):
        self.original,self.flat,self.tape,self.bounds=original,flat,tape,bounds
        self.arm=KINDS.index(kind)-1
        self.mask=ramp(original.device)
        self.checked=False
        self.calls=0
        self.changed=0
        self.diagnostics=[]
        self.masses=[]

    def wrap(self,actor,native):
        def forward(model,x):
            step=self.calls
            if (not model.training or not 0<=step<len(self.flat) or
                    not torch.equal(x,self.original[self.flat[step]])):
                raise ValueError("actual source update rows or order changed")
            self.calls+=1
            choices=self.tape[self.bounds[step]:self.bounds[step+1]]
            if not len(choices):
                return native(x)
            slots=torch.as_tensor(choices[:,1:3],device=x.device)
            directions=torch.as_tensor(choices[:,3],dtype=torch.long,device=x.device)
            shift=torch.stack((directions,-directions),dim=1).flatten()*SHIFT
            rows=x[slots.flatten()]
            mates=x[slots.flip(1).flatten()]
            if not self.checked:
                transforms(rows,mates,shift,self.mask,mass=0.)
                self.checked=True
            transformed,proof=transforms(rows,mates,shift,self.mask)
            current=x.clone()
            current[slots.flatten()]=transformed[self.arm]
            self.changed+=slots.numel()
            self.diagnostics.append(proof)
            self.masses.append(len(rows))
            return native(current)
        return MethodType(forward,actor)

    def report(self,want):
        if self.calls!=len(self.flat) or self.changed!=want or not self.checked:
            raise ValueError("unexercised or incomplete paired source augmentation")
        traces=torch.stack(self.diagnostics).cpu().numpy()
        mid=np.average(traces,weights=self.masses,axis=0)
        return dict(steps_applied=self.calls,changed_source_rows=self.changed,zero_dose_exact=True,
                    severity_error=float(traces[:,0].max()),cross_severity=float(mid[1]),same_severity=float(mid[2]),
                    cross_seam=float(mid[3]),same_seam=float(mid[4]),treatment_movement=float(traces[:,5].max()),
                    cross_fraction=float(mid[6]),same_fraction=float(mid[7]))
