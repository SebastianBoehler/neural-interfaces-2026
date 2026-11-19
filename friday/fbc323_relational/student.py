"""Fit original zero-feature S1C with true or deranged source relations."""
import argparse
from pathlib import Path

import numpy as np
import torch

from anchors import DATA_SHA, ORIGIN_SHA, PEOPLE, RUNTIME, row_sha, sha, split
from bank import TEACHER_SHA, load
from fit_relations import diagnostic, train
from matched_pairs import plan, to_device
from phase_source import F16_ACTOR, authenticated
from produce_distill import publish
from relational import BETA, HERE, KINDS, WEIGHT, ids, tag_for
from source_targets import normalizer


F14 = HERE.parent / "friday_s1c_training_20261118"
F16 = HERE.parent / "friday_s1c_distill_20261118"


def run(held, seed, smoke):
    from encoder312 import fingerprint
    import braindecode
    import model322
    from source_labels import read_training_labels

    data = F14/"proteus_windows.npz"
    if (sha(data)!=DATA_SHA or sha(F14/"runtime.lock")!=RUNTIME or sha(F14/"original/source323.sha256")!=ORIGIN_SHA or
            Path(model322.__file__).resolve()!=F14/"original/model322.py" or sha(F16/"actor.sha256")!=F16_ACTOR):
        raise ValueError("pinned student runtime or original waveform differs")
    device = torch.cuda.get_device_name()
    if "H100" not in device or braindecode.__version__ != "1.8.1":
        raise ValueError("student source requires exact H100 runtime")
    torch.set_num_threads(2)
    with np.load(data,allow_pickle=False) as z:
        waveform,subject,visit=(z[k] for k in ("X","subject","session"))
    people=ids(subject)
    stream=np.char.add(np.char.add(subject,"|"),visit)
    source, local_stream, roster=split(subject,visit,held)
    base,batches=authenticated(F16,F16,held,seed,smoke,source,local_stream,roster)
    cpu,gpu=torch.random.get_rng_state(),torch.cuda.get_rng_state()
    embedding,teachers=load(source,people,stream,held,seed,smoke,base)
    source_labels=read_training_labels(data,source,len(waveform))
    doses,picked,proof=plan(held,source,people[source],source_labels,batches,embedding)
    if not torch.equal(cpu,torch.random.get_rng_state()) or not torch.equal(gpu,torch.cuda.get_rng_state()):
        raise ValueError("frozen target pairing consumed model RNG")
    rows_input=dict(X=waveform[source],y=source_labels,stream=local_stream)
    x,y,scale,mu,sd,weights,prepared=model322.setup(rows_input,np.arange(len(source)),"cuda",epochs=base["epochs"])
    flat=[r for e in batches for r in e]
    if scale!=base["scale"] or any(not np.array_equal(a,b) for a,b in zip(flat,(r for e in prepared for r in e),strict=True)):
        raise ValueError("student original scale or batches differ")
    targets=to_device(doses,x.device)
    fixed=dict(held=held,seed=seed,epochs=base["epochs"],scale=scale,data=DATA_SHA,origin=ORIGIN_SHA,
               runtime=RUNTIME,device=device,actor=sha(HERE/"actor.sha256"),teacher_actor=TEACHER_SHA,teachers=teachers,
               source_rows=row_sha(source),source_people=roster,batches=base["batches"],initial=base["initial"],
               parameters=347903,weights_origin=base["checkpoint"],held_labels_opened=False,
               source_labels_only=True,rng_isolated=True,beta=BETA,q_gradients_nonzero=0,
               torch=str(torch.__version__),braindecode=braindecode.__version__,**proof)
    states=set()
    for kind in KINDS:
        model=model322.Factor("session",False,"cuda")
        normalizer(model)
        if (fingerprint(model)!=base["initial"] or sum(p.numel() for p in model.parameters() if p.requires_grad)!=347903):
            raise ValueError("student hard raw backbone initial state differs")
        q=[]
        hook=model.task.weight.register_hook(lambda g:q.append(g[:,-287:].count_nonzero()))
        try:
            if kind=="hard":
                fit=dict(model322.train(model,x,y,mu,sd,weights,batches),actual_pairs=0)
            else:
                fit=train(model,x,y,mu,sd,weights,batches,targets,kind,seed)
        finally:
            hook.remove()
        state=fingerprint(model)
        source_error=diagnostic(model,x,flat,targets,picked)
        if (state in states or state!=fingerprint(model) or state==base["initial"] or len(q)!=base["steps"] or
                torch.stack(q).sum().item() or fit["steps"]!=base["steps"] or
                (kind=="hard" and state!=base["state"]) or
                (kind!="hard" and fit["actual_pairs"]!=proof["pair_steps"])):
            raise ValueError("source fit did not match real optimizer, exact hard or relation work")
        states.add(state)
        out=HERE/f"{kind}_{tag_for(held,seed,smoke)}.pt"
        publish(out,model,dict(fixed,kind=kind,state=state,relation_weight=WEIGHT if kind!="hard" else 0.,
                               source_error=source_error,**fit))
        if kind=="hard" and sha(out)!=base["checkpoint"]:
            raise ValueError("exact hard F16 checkpoint bytes differ")
    print(f"PASS F28 paired source {tag_for(held,seed,smoke)} {proof['pair_count']} matched pairs",flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("held",choices=PEOPLE)
    cli.add_argument("seed",type=int,choices=(0,1))
    cli.add_argument("--smoke",action="store_true")
    a=cli.parse_args()
    run(a.held,a.seed,a.smoke)
