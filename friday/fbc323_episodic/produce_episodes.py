"""Fit exact hard CE and two matched training-only temporary-head students."""
import argparse
from pathlib import Path

import numpy as np
import torch

from anchors import DATA_SHA,ORIGIN_SHA,PEOPLE,RUNTIME,row_sha,sha,split
from episodes import ETA,HERE,KINDS,WEIGHT,plan
from phase_source import F16_ACTOR,authenticated
from produce_distill import publish
from source_targets import normalizer
from train_episodes import diagnostic,train


F14=HERE.parent/"friday_s1c_training_20261118"
F16=HERE.parent/"friday_s1c_distill_20261118"


def run(held,seed,smoke):
    from encoder312 import fingerprint
    import braindecode
    import model322
    from source_labels import read_training_labels

    path=F14/"proteus_windows.npz"
    if (sha(path)!=DATA_SHA or sha(F14/"runtime.lock")!=RUNTIME or
            sha(F14/"original/source323.sha256")!=ORIGIN_SHA or sha(F16/"actor.sha256")!=F16_ACTOR or
            Path(model322.__file__).resolve()!=F14/"original/model322.py"):
        raise ValueError("pinned raw tensor or native training actor differs")
    device=torch.cuda.get_device_name()
    if "H100" not in device or braindecode.__version__!="1.8.1":
        raise ValueError("requires exact H100 runtime")
    torch.set_num_threads(2)
    with np.load(path,allow_pickle=False) as z:
        eeg,subject,visit=(z[k] for k in ("X","subject","session"))
    source,stream,roster=split(subject,visit,held)
    base,batches=authenticated(F16,F16,held,seed,smoke,source,stream,roster)
    cpu,gpu=torch.random.get_rng_state(),torch.cuda.get_rng_state()
    labels=read_training_labels(path,source,len(eeg))
    planned,picked,proof=plan(source,subject[source],labels,batches)
    if not torch.equal(cpu,torch.random.get_rng_state()) or not torch.equal(gpu,torch.cuda.get_rng_state()):
        raise ValueError("prospective episode planning consumed model RNG")
    x,y,scale,mu,sd,weights,check=model322.setup(dict(X=eeg[source],y=labels,stream=stream),np.arange(len(source)),
                                                  "cuda",epochs=base["epochs"])
    flat=[r for e in batches for r in e]
    if (x.shape[1:]!=(41,480) or not torch.isfinite(x).all() or scale!=base["scale"] or
            any(not np.array_equal(a,b) for a,b in zip(flat,(r for e in check for r in e),strict=True))):
        raise ValueError("official source batch, tensor or scale differs")
    doses=[None if roles is None else tuple(torch.as_tensor(p,device=x.device) for p in roles) for roles in planned]
    fixed=dict(held=held,seed=seed,epochs=base["epochs"],scale=scale,data=DATA_SHA,origin=ORIGIN_SHA,
               runtime=RUNTIME,device=device,actor=sha(HERE/"actor.sha256"),initial=base["initial"],
               source_rows=row_sha(source),source_people=roster,batches=base["batches"],parameters=347903,
               weights_origin=base["checkpoint"],held_labels_opened=False,source_labels_only=True,rng_isolated=True,
               step_size=ETA,q_gradients_nonzero=0,torch=str(torch.__version__),braindecode=braindecode.__version__,**proof)
    tag=f"{'smoke_' if smoke else ''}{seed}_{held}"
    states=set()
    for kind in KINDS:
        model=model322.Factor("session",False,"cuda")
        normalizer(model)
        if (fingerprint(model)!=base["initial"] or sum(p.numel() for p in model.parameters() if p.requires_grad)!=347903):
            raise ValueError("raw student initialization or unchanged class-head count differs")
        q=[]
        head=model.task.weight.register_hook(lambda g:q.append(g[:,-287:].count_nonzero()))
        try:
            if kind=="hard":
                fit=dict(model322.train(model,x,y,mu,sd,weights,batches),actual_episodes=0)
            else:
                fit=train(model,x,y,mu,sd,weights,flat,doses,seed,kind)
        finally:
            head.remove()
        state=fingerprint(model)
        error=diagnostic(model,x,y,flat,doses,picked)
        if (state==fixed["initial"] or state in states or fingerprint(model)!=state or fit["steps"]!=base["steps"] or
                len(q)!=base["steps"] or torch.stack(q).sum().item() or
                kind=="hard" and state!=base["state"] or
                fit["actual_episodes"]!=(0 if kind=="hard" else proof["episode_steps"])):
            raise ValueError("native fit, temporary episodes or exact hard state differs")
        states.add(state)
        destination=HERE/f"{kind}_{tag}.pt"
        publish(destination,model,dict(fixed,kind=kind,state=state,source_error=error,
                                       episode_weight=0. if kind=="hard" else WEIGHT,**fit))
        if kind=="hard" and sha(destination)!=base["checkpoint"]:
            raise ValueError("exact parent checkpoint bytes differ")
    print(f"PASS F30 paired source {tag} {proof['episode_steps']} disjoint balanced episodes",flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("held",choices=PEOPLE)
    cli.add_argument("seed",type=int,choices=(0,1))
    cli.add_argument("--smoke",action="store_true")
    v=cli.parse_args()
    run(v.held,v.seed,v.smoke)
