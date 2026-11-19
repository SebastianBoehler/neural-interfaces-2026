"""Authenticate F32 source isolation and replay native actors with held labels closed."""
import argparse
import json
from math import log

import numpy as np
import torch

from anchors import DATA_SHA,ORIGIN_SHA,PEOPLE,RUNTIME,row_sha,sha,split
from auxiliary import PRIVATE_PARAMS
from external import ARMS,EVENT,F6_INPUTS,HERE,KINDS,MODES,PREFIX,TOL,WEIGHT,metadata,tape
from held import frozen_member
from held_distill import replay
from phase_source import F16_BANK,authenticated
from replay_f13 import DATA,ORDER_SHA,ORDERS,origin_report,raw_inputs


ACTOR="f8d074e6867ebfda82383c164c257d64f154b251b0ccdbcb5153b297e7917723"
F16=HERE.parent/"ni26_friday_s1c_distill_20261118"
EXTERNAL=HERE.parent/"neural-interfaces-2026/data/physionetmi"


def external_metadata():
    with np.load(DATA,allow_pickle=False) as z:
        return metadata(EXTERNAL,z["ch_names"])


def paired(held,seed,smoke,subject,visit,external):
    if sha(HERE/"actor.sha256")!=ACTOR or sha(F16/"before_labels.sha256")!=F16_BANK:
        raise ValueError("F32 actor or signed F16 blind bank differs")
    source,streams,people=split(subject,visit,held)
    base,batches=authenticated(F16,F16/"checkpoints",held,seed,smoke,source,streams,people)
    _,planned=tape(*external,held,seed,sum(map(len,batches)))
    tag=f"{'smoke_' if smoke else ''}{seed}_{held}"
    fixed=dict(held=held,seed=seed,epochs=base["epochs"],scale=base["scale"],data=DATA_SHA,origin=ORIGIN_SHA,
               runtime=RUNTIME,actor=ACTOR,source_rows=row_sha(source),source_people=people,batches=base["batches"],
               initial=base["initial"],parameters=347903,weights_origin=base["checkpoint"],steps=base["steps"],
               held_labels_opened=False,source_labels_only=True,rng_isolated=True,auxiliary_excluded=True,
               q_gradients_nonzero=0,torch=base["torch"],braindecode=base["braindecode"],external_inputs=F6_INPUTS,**planned)
    result={}
    for kind in KINDS:
        path=HERE/"checkpoints"/f"{kind}_{tag}.pt"
        meta=json.loads(path.with_suffix(".json").read_text())
        expected=dict(fixed,kind=kind,checkpoint=sha(path),external_weight=0. if kind=="hard" else WEIGHT)
        if (any(meta[k]!=v for k,v in expected.items()) or "H100" not in meta["device"] or
                not np.isfinite([meta[k] for k in ("first_loss","last_loss","max_gradient","external_scale")]).all() or
                min(meta["max_gradient"],meta["external_scale"])<=0 or meta["state"]==meta["initial"] or
                (kind!="joint")!=(meta["state"]==base["state"])):
            raise ValueError(f"matched source optimization or authentic detached state differs: {kind}")
        if kind=="hard":
            if meta["checkpoint"]!=base["checkpoint"] or meta["actual_aux"]:
                raise ValueError("exact hard source checkpoint differs")
        else:
            positive=("external_head_gradient","external_norm_gradient","external_final_loss","external_mean_loss")
            signals=[meta[k] for k in positive]
            body=meta["external_body_gradient"]
            if (not np.isfinite([body,*signals]).all() or min(signals)<=0 or
                    kind=="detached" and body!=0 or kind=="joint" and not body>1e-8 or
                    meta["actual_aux"]!=base["steps"] or meta["auxiliary_parameters"]!=PRIVATE_PARAMS or
                    meta["auxiliary_initial"]==meta["auxiliary_final"] or meta["native_norm_isolated"] is not True or
                    not 0<=meta["tail_drift"]<=TOL or not 0<meta["filter_radius"]<=EVENT-PREFIX or
                    not 0<=meta["external_final_accuracy"]<=1 or smoke and meta["external_final_loss"]>=log(5)):
                raise ValueError("external activity, timing or gradient-isolation checks differ")
        result[kind]=path,meta
    detached,joint=[result[k][1] for k in ("detached","joint")]
    if (detached["auxiliary_initial"]!=joint["auxiliary_initial"] or detached["filter_radius"]!=joint["filter_radius"] or
            len({(m["device"],m["external_scale"]) for _,m in result.values()})!=1):
        raise ValueError("external twins did not share initialization, source scale, geometry or platform")
    return result


def run(held,seed,smoke):
    torch.set_num_threads(4)
    origin_report()
    x,subject,person,stream,perms=raw_inputs()
    visit=np.asarray([v.rsplit("|",1)[-1] for v in stream])
    rows=np.flatnonzero(person==held)
    decisions,evidence={},{}
    for kind,(path,meta) in paired(held,seed,smoke,subject,visit,external_metadata()).items():
        bank,proof=replay(*frozen_member(path,meta),x,stream,rows,perms,meta["scale"])
        for mode,value in zip(MODES,bank,strict=True):
            decisions[f"{kind}_{mode}"]=value
        evidence[kind]=dict(proof,checkpoint=meta["checkpoint"],state=meta["state"],tape=meta["external_tape"],
                            receipt=sha(path.with_suffix(".json")))
    if (tuple(decisions)!=ARMS or not all(p["clips"] for p in evidence.values()) or
            any(not np.array_equal(decisions[f"hard_{m}"],decisions[f"detached_{m}"]) for m in MODES)):
        raise ValueError("causal controls are incomplete or literal detach changed held predictions")
    record=dict(held=held,seed=seed,actor=ACTOR,scorer=sha(HERE/"score.sha256"),data=DATA_SHA,
                origin=ORIGIN_SHA,orders=list(ORDERS),order_sha=ORDER_SHA,held_labels_opened=False,evidence=evidence)
    if not smoke:
        name=f"{held}_s{seed}.npz"
        with np.load(F16/"folds"/name,allow_pickle=False) as original:
            if not np.array_equal(original["index"],rows) or any(not np.array_equal(
                    original[f"hard_{m}"],decisions[f"hard_{m}"]) for m in MODES):
                raise ValueError("F32 hard held decisions differ from sealed F16")
        path=HERE/"folds"/name
        with path.open("xb") as out:
            np.savez_compressed(out,index=rows,metadata=json.dumps(record,sort_keys=True,allow_nan=False),**decisions)
        with path.with_suffix(".sha256").open("x") as out:
            out.write(sha(path)+"\n")
    print(json.dumps(record,sort_keys=True,allow_nan=False),flush=True)
    print(f"PASS F32 held {held} seed {seed}",flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("held",choices=PEOPLE)
    cli.add_argument("seed",type=int,choices=(0,1))
    cli.add_argument("--smoke",action="store_true")
    args=cli.parse_args()
    run(args.held,args.seed,args.smoke)
