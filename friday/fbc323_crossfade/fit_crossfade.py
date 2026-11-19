"""Fit matched source-only waveform crossfades using exact native A100 training."""
import argparse
import json
from pathlib import Path

import braindecode
import numpy as np
import torch

from anchors import DATA_SHA,ORIGIN_SHA,PEOPLE,row_sha,sha,split
from canary import model
from crossfade import Feed,HERE,KINDS,PARAMETERS
from encoder312 import fingerprint
from group import ACTOR as F33A
from group import paired as anchor
import model322
from pairs import audit,plan
from produce_distill import publish
from recipe import BD,CPU_CUDA,GPU,ROOT,TORCH,canary,historical,tag
from source_labels import read_training_labels


def run(held,seed,smoke):
    prior=canary()
    numeric=dict(cudnn=torch.backends.cudnn.version(),cudnn_tf32=torch.backends.cudnn.allow_tf32,
                 matmul_tf32=torch.backends.cuda.matmul.allow_tf32)
    if (torch.cuda.get_device_name()!=GPU or str(torch.__version__)!=TORCH or braindecode.__version__!=BD or
            Path(model322.__file__).resolve()!=ROOT/"model322.py" or any(numeric[k]!=prior[k] for k in CPU_CUDA)):
        raise ValueError("native A100 training environment differs")
    torch.set_num_threads(2)
    path=ROOT/"data/proteus_windows.npz"
    if sha(path)!=DATA_SHA:
        raise ValueError("wrong original raw waveform")
    with np.load(path,allow_pickle=False) as z:
        eeg,subject,visit=(z[k] for k in ("X","subject","session"))
    if eeg.shape!=(2655,41,480) or eeg.dtype!=np.float32 or not np.isfinite(eeg).all():
        raise ValueError("invalid official waveform shape or finite values")
    source,streams,people=split(subject,visit,held)
    base,batches,_=historical(held,seed,smoke,source,streams,people)
    reference=anchor(held,seed,smoke,subject,visit)["hard"][1]
    labels=read_training_labels(path,source,len(eeg))
    rng=(torch.random.get_rng_state(),torch.cuda.get_rng_state())
    tape=plan(batches,streams,labels,held,seed)
    bounds,paired=audit(batches,streams,tape,labels)
    if not torch.equal(rng[0],torch.random.get_rng_state()) or not torch.equal(rng[1],torch.cuda.get_rng_state()):
        raise ValueError("source pairing changed native RNG")
    x,y,scale,mu,sd,weights,actual=model322.setup(dict(X=eeg[source],y=labels,stream=streams),
                                                   np.arange(len(source)),"cuda",epochs=base["epochs"])
    flat=[row for epoch in batches for row in epoch]
    if scale!=reference["scale"] or any(not np.array_equal(p,q) for p,q in
                                       zip(flat,(r for epoch in actual for r in epoch),strict=True)):
        raise ValueError("native source scale or update schedule differed")
    fixed=dict(held=held,seed=seed,epochs=base["epochs"],scale=scale,initial=base["initial"],data=DATA_SHA,
               origin=ORIGIN_SHA,source_rows=row_sha(source),source_people=people,batches=base["batches"],
               held_labels_opened=False,actor=sha(HERE/"actor.sha256"),f33a=F33A,weights_origin=reference["checkpoint"],
               device=GPU,torch=TORCH,braindecode=BD,rng_isolated=True,parameters=PARAMETERS,q_gradients_nonzero=0,
               **numeric,**paired)
    name=tag(held,seed,smoke)
    box=HERE/"checkpoints"
    receipt=box/f"pairs_{name}.npz"
    with receipt.open("xb") as out:
        np.savez_compressed(out,pairs=tape,metadata=json.dumps(fixed,sort_keys=True,allow_nan=False))
    lock=sha(receipt)
    states=set()
    for kind in KINDS:
        actor=model(seed)
        if fingerprint(actor)!=reference["initial"]:
            raise ValueError("exact A100 source initialization differs")
        native=actor.forward
        feed=Feed(x,flat,tape,bounds,kind) if kind!="hard" else None
        if feed is not None:
            actor.forward=feed.wrap(actor,native)
        q=[]
        hook=actor.task.weight.register_hook(lambda g:q.append(g[:,-287:].count_nonzero()))
        try:
            stats=model322.train(actor,x,y,mu,sd,weights,batches)
        finally:
            hook.remove()
            actor.forward=native
        state=fingerprint(actor)
        if (stats["steps"]!=base["steps"] or len(q)!=base["steps"] or torch.stack(q).sum().item() or
                state in states or state==reference["initial"] or sha(receipt)!=lock):
            raise ValueError("original source update, fixed-zero gradient or fitted state differed")
        proof={} if feed is None else feed.report(paired["treated_rows"])
        states.add(state)
        target=box/f"{kind}_{name}.pt"
        publish(target,actor,dict(fixed,kind=kind,state=state,tape=lock,**stats,**proof))
        if kind=="hard" and (state!=reference["state"] or sha(target)!=reference["checkpoint"]):
            raise ValueError("true native hard A100 checkpoint bytes differed")
    print(f"PASS F36 paired source {name}",flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("held",choices=PEOPLE)
    cli.add_argument("seed",type=int,choices=(0,1))
    cli.add_argument("--smoke",action="store_true")
    args=cli.parse_args()
    run(args.held,args.seed,args.smoke)
