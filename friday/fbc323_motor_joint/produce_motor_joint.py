"""Fit exact, detached, and joint raw S1C source actors on Ferranti."""
import argparse
from math import log
from pathlib import Path

import numpy as np
import torch

from anchors import DATA_SHA,ORIGIN_SHA,PEOPLE,RUNTIME,row_sha,sha,split
from auxiliary import MotorHead,PRIVATE_PARAMS,diagnostic
from external import F6_INPUTS,HERE,KINDS,WEIGHT,metadata,tape,waveforms
from phase_source import F16_ACTOR,authenticated
from produce_distill import publish
from source_targets import normalizer
from train_motor import train


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
        eeg,subject,visit,channels=(z[k] for k in ("X","subject","session","ch_names"))
    source,stream,roster=split(subject,visit,held)
    base,batches=authenticated(F16,F16,held,seed,smoke,source,stream,roster)
    flat=[rows for epoch in batches for rows in epoch]
    cpu,gpu=torch.random.get_rng_state(),torch.cuda.get_rng_state()
    labels=read_training_labels(path,source,len(eeg))
    person,fy=metadata(HERE/"physionet",channels)
    indices,proof=tape(person,fy,held,seed,len(flat))
    fx,external_scale=waveforms(HERE/"physionet")
    if not torch.equal(cpu,torch.random.get_rng_state()) or not torch.equal(gpu,torch.cuda.get_rng_state()):
        raise ValueError("external preparation consumed model RNG")
    fy=torch.as_tensor(fy,dtype=torch.long,device=fx.device)
    indices=torch.as_tensor(indices,device=fx.device)
    x,y,scale,mu,sd,weights,check=model322.setup(dict(X=eeg[source],y=labels,stream=stream),np.arange(len(source)),
                                               "cuda",epochs=base["epochs"])
    if (x.shape[1:]!=(41,480) or not torch.isfinite(x).all() or scale!=base["scale"] or
            any(not np.array_equal(a,b) for a,b in zip(flat,(r for e in check for r in e),strict=True))):
        raise ValueError("official source batch, tensor or scale differs")
    fixed=dict(held=held,seed=seed,epochs=base["epochs"],scale=scale,data=DATA_SHA,origin=ORIGIN_SHA,
               runtime=RUNTIME,device=device,actor=sha(HERE/"actor.sha256"),initial=base["initial"],
               source_rows=row_sha(source),source_people=roster,batches=base["batches"],parameters=347903,
               weights_origin=base["checkpoint"],held_labels_opened=False,source_labels_only=True,rng_isolated=True,
               auxiliary_excluded=True,q_gradients_nonzero=0,torch=str(torch.__version__),braindecode=braindecode.__version__,
               external_scale=external_scale,external_inputs=F6_INPUTS,**proof)
    tag=f"{'smoke_' if smoke else ''}{seed}_{held}"
    auxiliary_initial=None
    checks={}
    for kind in KINDS:
        model=model322.Factor("session",False,"cuda")
        normalizer(model)
        if (fingerprint(model)!=base["initial"] or sum(p.numel() for p in model.parameters() if p.requires_grad)!=347903):
            raise ValueError("raw student initialization or parameter count differs")
        head=None
        if kind!="hard":
            cpu,gpu=torch.random.get_rng_state(),torch.cuda.get_rng_state()
            head=MotorHead(seed,x.device)
            start=fingerprint(head)
            if (not torch.equal(cpu,torch.random.get_rng_state()) or not torch.equal(gpu,torch.cuda.get_rng_state())
                    or sum(p.numel() for p in head.parameters())!=PRIVATE_PARAMS
                    or auxiliary_initial is not None and start!=auxiliary_initial):
                raise ValueError("private heads changed native RNG, initialization or parameter count")
            auxiliary_initial=start
        q=[]
        handle=model.task.weight.register_hook(lambda g:q.append(g[:,-287:].count_nonzero()))
        try:
            if kind=="hard":
                fit=dict(model322.train(model,x,y,mu,sd,weights,batches),actual_aux=0)
            else:
                fit=train(model,head,x,y,mu,sd,weights,flat,fx,fy,indices,seed,kind=="detached")
        finally:
            handle.remove()
        state=fingerprint(model)
        extra={}
        if head is not None:
            extra=diagnostic(model,head,fx,fy,indices)
            checks[kind]=extra["external_final_loss"]
            extra.update(auxiliary_initial=auxiliary_initial,auxiliary_final=fingerprint(head),
                         auxiliary_parameters=PRIVATE_PARAMS)
            if extra["auxiliary_final"]==auxiliary_initial:
                raise ValueError("private auxiliary failed to train")
        if (state==fixed["initial"] or fingerprint(model)!=state or fit["steps"]!=base["steps"] or
                len(q)!=base["steps"] or torch.stack(q).sum().item() or
                (kind!="joint")!=(state==base["state"]) or
                fit["actual_aux"]!=(0 if kind=="hard" else base["steps"])):
            raise ValueError("original, detached, or joint native state differs from registration")
        destination=HERE/f"{kind}_{tag}.pt"
        publish(destination,model,dict(fixed,kind=kind,state=state,external_weight=0. if kind=="hard" else WEIGHT,
                                       **fit,**extra))
        if kind=="hard" and sha(destination)!=base["checkpoint"]:
            raise ValueError("exact parent checkpoint bytes differ")
    if smoke and any(loss>=log(5) for loss in checks.values()):
        raise ValueError(f"real private motor heads did not learn above chance: {checks}")
    print(f"PASS F32 paired source {tag} private motor CE {checks}",flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("held",choices=PEOPLE)
    cli.add_argument("seed",type=int,choices=(0,1))
    cli.add_argument("--smoke",action="store_true")
    v=cli.parse_args()
    run(v.held,v.seed,v.smoke)
