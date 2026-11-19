"""Train the fixed F34 envelope models against the exact F33a A100 anchor."""
import argparse
from pathlib import Path

import braindecode
import numpy as np
import torch

from anchors import DATA_SHA,ORIGIN_SHA,PEOPLE,row_sha,sha,split
from canary import median_time,model
from checks import evidence,fit,install
from encoder312 import fingerprint
from envelope import HERE,KINDS,MAX_RATIO,MAX_RESERVED,trainable
from group import ACTOR as F33A
from group import paired as anchor
import model322
from produce_distill import publish
from recipe import BD,CPU_CUDA,GPU,ROOT,TORCH,canary,historical,tag
from source_labels import read_training_labels


def run(held,seed,smoke):
    device=torch.cuda.get_device_name()
    witness=canary()
    numeric=dict(cudnn=torch.backends.cudnn.version(),cudnn_tf32=torch.backends.cudnn.allow_tf32,
                 matmul_tf32=torch.backends.cuda.matmul.allow_tf32)
    if (device!=GPU or str(torch.__version__)!=TORCH or braindecode.__version__!=BD or
            Path(model322.__file__).resolve()!=ROOT/"model322.py" or
            any(numeric[k]!=witness[k] for k in CPU_CUDA)):
        raise ValueError("fixed A100 numerical path differs")
    torch.set_num_threads(2)
    p=ROOT/"data/proteus_windows.npz"
    if sha(p)!=DATA_SHA:
        raise ValueError("original waveform differs")
    with np.load(p,allow_pickle=False) as z:
        eeg,subject,visit=(z[k] for k in ("X","subject","session"))
    if eeg.dtype!=np.float32 or eeg.shape!=(2655,41,480) or not np.isfinite(eeg).all():
        raise ValueError("unexpected official raw waveform shape or values")
    source,streams,people=split(subject,visit,held)
    base,batches,_=historical(held,seed,smoke,source,streams,people)
    reference=anchor(held,seed,smoke,subject,visit)["hard"][1]
    labels=read_training_labels(p,source,len(eeg))
    x,y,scale,mu,sd,weights,observed=model322.setup(dict(X=eeg[source],y=labels,stream=streams),
                                                    np.arange(len(source)),"cuda",epochs=base["epochs"])
    flat=[r for ep in batches for r in ep]
    if (scale!=reference["scale"] or any(not np.array_equal(a,b) for a,b in
                                       zip(flat,(r for ep in observed for r in ep),strict=True))):
        raise ValueError("native A100 scale or source batches changed")
    fit_args=x,y,mu,sd,weights,batches
    sample=x[flat[0][:32]]
    fixed=dict(held=held,seed=seed,epochs=base["epochs"],scale=scale,initial=base["initial"],data=DATA_SHA,
               origin=ORIGIN_SHA,source_rows=row_sha(source),source_people=people,batches=base["batches"],
               held_labels_opened=False,actor=sha(HERE/"actor.sha256"),f33a=F33A,weights_origin=reference["checkpoint"],
               device=device,torch=TORCH,braindecode=BD,rng_isolated=True,zero_identity=True,
               q_gradients_nonzero=0,diagnostic_rows=row_sha(source[flat[0][:32]]),**numeric)
    states=set()
    initial_branch=None
    hard_ms=None
    name=tag(held,seed,smoke)
    for kind in KINDS:
        actor=model(seed)
        if fingerprint(actor)!=reference["initial"]:
            raise ValueError("A100 native initial state differs")
        branch,proof=install(actor,kind,seed,sample)
        stats=fit(actor,branch,fit_args,reference)
        if branch is not None:
            if initial_branch is not None and proof["initial_branch"]!=initial_branch:
                raise ValueError("matched waveform branches changed paired initialization")
            initial_branch=proof["initial_branch"]
            proof.update(evidence(actor,branch,sample))
        state=stats["state"]
        if state in states or fingerprint(actor)!=state:
            raise ValueError("duplicate or mutated fitted waveform state")
        states.add(state)
        peak=torch.cuda.max_memory_reserved()
        if smoke:
            ms=median_time(actor,sample)
            hard_ms=ms if kind=="hard" else hard_ms
            if not 0<ms<=MAX_RATIO*hard_ms or peak>MAX_RESERVED:
                raise ValueError("matched warm timing or CUDA reserve did not qualify")
            proof.update(warm_ms=ms,hard_ms=hard_ms)
        target=HERE/"checkpoints"/f"{kind}_{name}.pt"
        publish(target,actor,dict(fixed,kind=kind,parameters=trainable(actor),peak_reserved=peak,**stats,**proof))
        if kind=="hard" and sha(target)!=reference["checkpoint"]:
            raise ValueError("original exact A100 native hard checkpoint bytes differ")
    print(f"PASS F34 paired source {name}",flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("held",choices=PEOPLE)
    cli.add_argument("seed",type=int,choices=(0,1))
    cli.add_argument("--smoke",action="store_true")
    args=cli.parse_args()
    run(args.held,args.seed,args.smoke)
