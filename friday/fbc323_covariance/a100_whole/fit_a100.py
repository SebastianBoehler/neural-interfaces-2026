"""Fit each A100 matched F33a fold and its independently derived hard reference."""
import argparse
from pathlib import Path

import braindecode
import numpy as np
import torch

from anchors import DATA_SHA,ORIGIN_SHA,PEOPLE,row_sha,sha,split
from canary import median_time,model
from covariance import KINDS,MAX_RATIO,MAX_RESERVED,order,trainable
from encoder312 import fingerprint
import model322
from produce_distill import publish
from recipe import BD,CANARY_SHA,CPU_CUDA,F33,GPU,HERE,ROOT,TORCH,canary,historical,tag
from source_checks import diagnostics,fit,install
from source_labels import read_training_labels


def run(held,seed,smoke):
    device=torch.cuda.get_device_name()
    expected=canary()
    numeric=dict(cudnn=torch.backends.cudnn.version(),cudnn_tf32=torch.backends.cudnn.allow_tf32,
                 matmul_tf32=torch.backends.cuda.matmul.allow_tf32)
    if (device!=GPU or str(torch.__version__)!=TORCH or braindecode.__version__!=BD or
            Path(model322.__file__).resolve()!=ROOT/"model322.py" or
            any(numeric[k]!=expected[k] for k in CPU_CUDA)):
        raise ValueError("A100 hardware, runtime or numeric flags differ from canary")
    torch.set_num_threads(2)
    path=ROOT/"data/proteus_windows.npz"
    if sha(path)!=DATA_SHA:
        raise ValueError("original raw tensor changed")
    with np.load(path,allow_pickle=False) as z:
        eeg,subject,visit=(z[k] for k in ("X","subject","session"))
    if eeg.shape!=(2655,41,480) or eeg.dtype!=np.float32 or not np.isfinite(eeg).all():
        raise ValueError("wrong official EEG shape")
    source,streams,people=split(subject,visit,held)
    reference,batches,receipt=historical(held,seed,smoke,source,streams,people)
    labels=read_training_labels(path,source,len(eeg))
    x,y,scale,mu,sd,weight,observed=model322.setup(dict(X=eeg[source],y=labels,stream=streams),
                                                   np.arange(len(source)),"cuda",epochs=reference["epochs"])
    flat=[rows for ep in batches for rows in ep]
    if (not np.isclose(scale,reference["scale"],atol=0,rtol=2e-6) or
            any(not np.array_equal(p,q) for p,q in zip(flat,(r for ep in observed for r in ep),strict=True))):
        raise ValueError("A100 source batching or normalized scale differs")
    args=x,y,mu,sd,weight,batches
    bare=model(seed)
    initial=fingerprint(bare)
    if initial!=reference["initial"]:
        raise ValueError("original hard initialization differs")
    witness=model322.train(bare,*args)
    state=fingerprint(bare)
    if (witness["steps"]!=reference["steps"] or state==initial or
            smoke and (state!=expected["a100_state"] or scale!=expected["scale"])):
        raise ValueError("independent hard fit did not reproduce the A100 canary or expected work")
    paired=dict(reference,state=state)
    common=dict(held=held,seed=seed,epochs=reference["epochs"],scale=scale,initial=initial,data=DATA_SHA,
                origin=ORIGIN_SHA,source_rows=row_sha(source),source_people=people,batches=reference["batches"],
                historical_metadata=receipt,h100_state=reference["state"],h100_weights_loaded=False,
                held_labels_opened=False,actor=sha(HERE/"actor.sha256"),f33=F33,canary=CANARY_SHA,
                device=device,torch=TORCH,braindecode=BD,**numeric)
    name=tag(held,seed,smoke)
    destination=HERE/"checkpoints"
    bare_path=destination/f"bare_{name}.pt"
    publish(bare_path,bare,dict(common,kind="hard",role="bare",parameters=trainable(bare),state=state,**witness))
    hard_base=sha(bare_path)
    del bare
    sample=x[flat[0][:32]]
    states=set()
    initial_basis=None
    hard_ms=None
    for kind in KINDS:
        actor=model(seed)
        if fingerprint(actor)!=initial:
            raise ValueError("same-device fit lost the original initialization")
        layer,proof=install(actor,kind,seed,sample)
        result=fit(actor,layer,args,paired)
        if layer is not None:
            if initial_basis is not None and proof["initial_basis"]!=initial_basis:
                raise ValueError("aligned and permuted use differing initial bases")
            initial_basis=proof["initial_basis"]
            proof.update(diagnostics(actor,layer,sample,kind),permutation=row_sha(order(kind)))
        if result["state"] in states or fingerprint(actor)!=result["state"]:
            raise ValueError("duplicate or modified source states")
        states.add(result["state"])
        peak=torch.cuda.max_memory_reserved()
        if smoke:
            ms=median_time(actor,sample)
            hard_ms=ms if kind=="hard" else hard_ms
            if not 0<ms<=MAX_RATIO*hard_ms or peak>MAX_RESERVED:
                raise ValueError("paired GPU timing or reservation budget failed")
            proof.update(warm_ms=ms,hard_ms=hard_ms)
        path=destination/f"{kind}_{name}.pt"
        publish(path,actor,dict(common,kind=kind,role="matched",bare_checkpoint=hard_base,
                                bare_state=state,train_rows_seen=sum(map(len,flat)),diagnostic_rows=row_sha(source[flat[0][:32]]),
                                rng_isolated=True,zero_identity=True,q_gradients_nonzero=0,
                                parameters=trainable(actor),**result,peak_reserved=peak,**proof))
        if kind=="hard" and sha(path)!=hard_base:
            raise ValueError("same-device bare and matched checkpoint bytes differ")
    print(f"PASS F33a paired source {name}",flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("held",choices=PEOPLE)
    cli.add_argument("seed",type=int,choices=(0,1))
    cli.add_argument("--smoke",action="store_true")
    args=cli.parse_args()
    run(args.held,args.seed,args.smoke)
