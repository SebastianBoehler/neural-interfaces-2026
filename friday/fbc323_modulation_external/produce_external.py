"""Fit same-device matched all-source F25 operators on Galvani A100."""
import argparse
import hashlib
from io import BytesIO
from pathlib import Path

import braindecode
import numpy as np
import torch

from anchors import DATA_SHA,ORIGIN_SHA,RUNTIME,row_sha,sha
from encoder312 import fingerprint
import model322
import probe
from modulation import BASE,KINDS,MAX_RATIO,MAX_RESERVED,architecture,trainable
from produce_distill import publish
from source_labels import read_training_labels
from source_probe import timing
from source_scope import authenticated,scope
from source_targets import normalizer


HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/"ni26_s323_normsplit_20261117"
F25=HERE.parent/"ni26_friday_s1c_modulation_20261119"
F16E=HERE.parent/"ni26_friday_s1c_motor_external_cpu_20261119/anchors_f16e"
F25_ACTOR="589b9a212cafdb0367f9f34bb840db5ff95c98ccfeea3bf77e268aa3ddb2e95c"


def native(seed):
    model322.SEED=seed
    actor=model322.Factor("session",False,"cuda")
    normalizer(actor)
    if trainable(actor)!=BASE:
        raise ValueError("original raw all-source architecture differs")
    return actor


def run(seed,smoke):
    device=torch.cuda.get_device_name()
    if (device!="NVIDIA A100-PCIE-40GB" or str(torch.__version__)!="2.14.1+cu126" or
            braindecode.__version__!="1.8.1" or Path(probe.__file__).resolve()!=F25/"probe.py" or
            Path(model322.__file__).resolve()!=ROOT/"model322.py" or sha(F25/"actor.sha256")!=F25_ACTOR):
        raise ValueError("source GPU, runtime or pinned original F25 implementation differs")
    torch.set_num_threads(2)
    path=ROOT/"data/proteus_windows.npz"
    x,rows,stream,roster,_=scope(path)
    ref,batches=authenticated(F16E,F16E,seed,smoke,rows,stream,roster)
    y=read_training_labels(path,rows,len(rows))
    x,y,scale,mu,sd,weights,actual=model322.setup(dict(X=x,y=y,stream=stream),rows,"cuda",epochs=ref["epochs"])
    flat=[r for ep in batches for r in ep]
    if (not np.isclose(scale,ref["scale"],atol=0,rtol=2e-6) or
            any(not np.array_equal(a,b) for a,b in zip(flat,(r for ep in actual for r in ep),strict=True))):
        raise ValueError("A100 source schedule or relative scale mismatch")
    args=x,y,mu,sd,weights,batches
    bare=native(seed)
    initial=fingerprint(bare)
    if initial!=ref["initial"]:
        raise ValueError("A100 initialization differs from the pinned F16e state")
    original=model322.train(bare,*args)
    state=fingerprint(bare)
    if state==initial or original["steps"]!=ref["steps"]:
        raise ValueError("same-device hard anchor did not fit the prescribed schedule")
    archive=BytesIO()
    torch.save(dict(kind="hard",held="all_source",seed=seed,scale=scale,state=bare.cpu().state_dict()),archive)
    hard_bytes=hashlib.sha256(archive.getbuffer()).hexdigest()
    paired=dict(ref,state=state)
    del bare,archive
    sample=x[flat[0][:32]]
    fixed=dict(held="all_source",seed=seed,epochs=ref["epochs"],scale=scale,dataset=DATA_SHA,source_people=roster,
               origin=ORIGIN_SHA,source_rows=row_sha(rows),source_count=len(rows),source_sessions=len(np.unique(stream)),
               source_only_labels=True,held_labels_opened=False,scherer_exposed=False,weights_origin=ref["checkpoint"],
               h100_state=ref["state"],runtime_reference=RUNTIME,initial=initial,batches=ref["batches"],
               device=device,torch=str(torch.__version__),braindecode=braindecode.__version__,
               cudnn=torch.backends.cudnn.version(),cudnn_tf32=torch.backends.cudnn.allow_tf32,
               matmul_tf32=torch.backends.cuda.matmul.allow_tf32,actor=sha(HERE/"actor.sha256"),f25=F25_ACTOR,
               diagnostic_rows=row_sha(rows[flat[0][:32]]),q_gradients_nonzero=0,zero_identity=True,rng_isolated=True,
               same_device_hard_state=state,same_device_hard_checkpoint=hard_bytes)
    states=set()
    hard_ms=None
    tag=f"{'smoke_' if smoke else ''}{seed}_all_source"
    for kind in KINDS:
        model=native(seed)
        if fingerprint(model)!=initial:
            raise ValueError("arms changed paired native initialization")
        layer=probe.install(model,kind,sample)
        fitted=probe.fit(model,layer,args,paired)
        proof={} if layer is None else probe.diagnostics(model,layer,sample,kind)
        ms=timing(model,sample)
        hard_ms=ms if kind=="hard" else hard_ms
        peak=torch.cuda.max_memory_reserved()
        if (not 0<ms<=MAX_RATIO*hard_ms or peak>MAX_RESERVED or fitted["state"] in states or
                fingerprint(model)!=fitted["state"]):
            raise ValueError("F25 paired source timing, distinct state or memory gate failed")
        states.add(fitted["state"])
        output=HERE/"checkpoints"/f"{kind}_{tag}.pt"
        publish(output,model,dict(fixed,kind=kind,warm_ms=ms,pinned_hard_ms=hard_ms,peak_reserved=peak,
                                  **architecture(kind),**fitted,**proof))
        if kind=="hard" and sha(output)!=hard_bytes:
            raise ValueError("same-device bare and matched hard checkpoints differ")
    print(f"PASS F25e A100 paired source {tag}",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed",type=int,choices=(0,1))
    parser.add_argument("--smoke",action="store_true")
    v=parser.parse_args()
    run(v.seed,v.smoke)
