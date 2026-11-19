"""Fit exact native and two matched all-source F35e reweighting arms."""
import argparse
import json
from pathlib import Path

import braindecode
import numpy as np
import torch

from anchors import DATA_SHA,ORIGIN_SHA,row_sha,sha
from all_source import BUNDLE,F35,HERE,scope,source_anchor,tag
from canary import model
from contributions import KINDS,PARAMETERS,Survey,eligibility,imbalance,transfer
from encoder312 import fingerprint
import model322
from produce_distill import publish
from recipe import BD,CPU_CUDA,GPU,ROOT,TORCH,canary
from source_labels import read_training_labels
from weighted_train import train as weighted


def archive(path,trace,eligible,applied,receipt):
    with path.open("xb") as out:
        np.savez_compressed(out,trace=trace,eligible=eligible,applied=applied,
                            metadata=json.dumps(receipt,sort_keys=True,allow_nan=False))
    return sha(path)


def run(seed,smoke):
    approved=canary()
    numeric=dict(cudnn=torch.backends.cudnn.version(),cudnn_tf32=torch.backends.cudnn.allow_tf32,
                 matmul_tf32=torch.backends.cuda.matmul.allow_tf32)
    if (torch.cuda.get_device_name()!=GPU or str(torch.__version__)!=TORCH or braindecode.__version__!=BD or
            Path(model322.__file__).resolve()!=ROOT/"model322.py" or any(numeric[k]!=approved[k] for k in CPU_CUDA)):
        raise ValueError("exact A100 native platform differs")
    torch.set_num_threads(2)
    path=ROOT/"data/proteus_windows.npz"
    eeg,source,streams,people,_=scope(path)
    base,batches,reference=source_anchor(seed,smoke,source,streams,people)
    labels=read_training_labels(path,source,len(eeg))
    x,y,scale,mu,sd,weights,actual=model322.setup(dict(X=eeg,y=labels,stream=streams),
                                                   source,"cuda",epochs=base["epochs"])
    flat=[rows for ep in batches for rows in ep]
    if (scale!=reference["scale"] or any(not np.array_equal(a,b) for a,b in
                                       zip(flat,(r for ep in actual for r in ep),strict=True))):
        raise ValueError("original scale or source updates differ")
    eligible=eligibility(flat,streams)
    actor=model(seed)
    initial=fingerprint(actor)
    if initial!=reference["initial"]:
        raise ValueError("exact original native initial state differs")
    survey=Survey(actor,x,y,weights,flat,eligible,probe=True)
    q=[]
    hook=actor.task.weight.register_hook(lambda g:q.append(g[:,-287:].count_nonzero()))
    try:
        baseline=model322.train(actor,x,y,mu,sd,weights,batches)
    finally:
        survey.detach()
        hook.remove()
    if (fingerprint(actor)!=reference["state"] or baseline["steps"]!=base["steps"] or
            len(q)!=base["steps"] or torch.stack(q).sum().item()):
        raise ValueError("diagnostic changed native hard reference or update trace")
    trace=survey.proof()
    offsets=transfer(trace,eligible)
    zero=np.zeros_like(offsets)
    native,_=imbalance(trace,eligible,zero)
    balanced,_=imbalance(trace,eligible,offsets)
    mirrored,_=imbalance(trace,eligible,-offsets)
    shift=1-balanced/native
    if not shift>=.10 or not mirrored>native:
        raise ValueError("native-anchor contribution transfer is inactive")
    name=tag(seed,smoke)
    fixed=dict(held="all_source",seed=seed,epochs=base["epochs"],scale=scale,initial=initial,data=DATA_SHA,origin=ORIGIN_SHA,
               source_rows=row_sha(source),source_people=people,batches=base["batches"],held_labels_opened=False,
               actor=sha(HERE/"actor.sha256"),f25e=BUNDLE,weights_origin=reference["checkpoint"],device=GPU,
               f35=F35,source_count=2655,source_sessions=14,scherer_exposed=False,h100_weights_loaded=False,
               torch=TORCH,braindecode=BD,parameters=PARAMETERS,q_gradients_nonzero=0,updates_order_exact=True,
               **numeric)
    box=HERE/"checkpoints"
    tape=box/f"tape_{name}.npz"
    lock=archive(tape,trace,eligible,offsets,dict(fixed,hard_state=reference["state"],hard_gradient_error=survey.error,
                                                anchor_original=native,anchor_balanced=balanced,anchor_mirrored=mirrored))
    offsets.flags.writeable=False
    hard=box/f"hard_{name}.pt"
    publish(hard,actor,dict(fixed,kind="hard",state=reference["state"],tape=lock,hard_gradient_error=survey.error,
                           anchor_original=native,anchor_balanced=balanced,anchor_mirrored=mirrored,
                           anchor_relative_reduction=shift,eligible=int(eligible.sum()),**baseline))
    if sha(hard)!=reference["checkpoint"]:
        raise ValueError("hard bytes differ from original A100 decoder")
    if smoke:
        control=model(seed)
        control_stats,applied=weighted(control,x,y,mu,sd,weights,flat,zero,seed)
        if control_stats!=baseline or any(applied) or fingerprint(control)!=reference["state"]:
            raise ValueError("zero-transfer source trainer does not reproduce exact native optimization")
    states={reference["state"]}
    for kind,sign in zip(KINDS[1:],(1,-1),strict=True):
        actor=model(seed)
        if fingerprint(actor)!=initial:
            raise ValueError("paired source branch changed initialization")
        act=Survey(actor,x,y,weights,flat,eligible)
        q=[]
        handle=actor.task.weight.register_hook(lambda g:q.append(g[:,-287:].count_nonzero()))
        try:
            stats,used=weighted(actor,x,y,mu,sd,weights,flat,offsets*sign,seed)
        finally:
            act.detach()
            handle.remove()
        path=box/f"{kind}_{name}.pt"
        state=fingerprint(actor)
        used=np.asarray(used,dtype=np.float64)
        if (sha(tape)!=lock or not np.array_equal(used,offsets*sign) or stats["steps"]!=base["steps"] or
                len(q)!=base["steps"] or torch.stack(q).sum().item() or state in states):
            raise ValueError("paired source readouts changed tape, update count or distinct trained state")
        states.add(state)
        steps=act.proof()
        before,relative_before=imbalance(steps,eligible,zero)
        after,relative_after=imbalance(steps,eligible,used)
        live=archive(box/f"{kind}_trajectory_{name}.npz",steps,eligible,used,
                     dict(fixed,kind=kind,state=state,tape=lock))
        publish(path,actor,dict(fixed,kind=kind,state=state,tape=lock,trajectory=live,sign=sign,eligible=int(eligible.sum()),
                                actual_original=before,actual_applied=after,actual_relative_original=relative_before,
                                actual_relative_applied=relative_after,**stats))
    print(f"PASS F35e paired source {name} frozen contribution reduction {shift:.6f}",flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("seed",type=int,choices=(0,1))
    cli.add_argument("--smoke",action="store_true")
    args=cli.parse_args()
    run(args.seed,args.smoke)
