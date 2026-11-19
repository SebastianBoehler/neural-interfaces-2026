"""Publish bounded native artifacts based only on public source waveforms."""
import argparse
import json
from math import log

import numpy as np
import torch

from anchors import DATA_SHA,ORIGIN_SHA,RUNTIME,row_sha,sha
from export_checks import check,reference,reset
from external import EVENT,F6_INPUTS,KINDS,PREFIX,TOL,WEIGHT
from held import frozen_member
from incremental_model import export_actor
from replay_f13 import DATA,origin_report
from source_scope import F16E_LOCK,F32_ACTOR,HERE,authenticated,scope


ACTOR="fdbc95d3649c0a55efb1c4de6fdcfcb19b8d278a6c319392a7e3872f1d90417e"
F16E=HERE/"anchors_f16e"


def frozen_sources(seed,smoke,source,streams,people):
    if sha(HERE/"actor.sha256")!=ACTOR or sha(F16E/"before_labels.json")!=F16E_LOCK:
        raise ValueError("trained actor or sealed baseline lock differs")
    parent,_=authenticated(F16E,F16E,seed,smoke,source,streams,people)
    if not smoke:
        signed=json.loads((F16E/"before_labels.json").read_text())["source"][f"s{seed}_hard"]
        if any(signed[k]!=parent[k] for k in ("checkpoint","state")):
            raise ValueError("plain F16e state changed since its held lock")
    tag=f"{'smoke_' if smoke else ''}{seed}_all_source"
    fixed=dict(held="all_source",seed=seed,actor=ACTOR,epochs=parent["epochs"],steps=parent["steps"],
               runtime=RUNTIME,dataset=DATA_SHA,origin=ORIGIN_SHA,source_rows=row_sha(source),source_people=people,
               source_count=2655,source_sessions=14,batches=parent["batches"],scale=parent["scale"],
               initial=parent["initial"],parameters=347903,weights_origin=parent["checkpoint"],scherer_exposed=False,
               held_labels_opened=False,source_labels_only=True,q_gradients_nonzero=0,rng_isolated=True,
               auxiliary_excluded=True,parent_objective=F32_ACTOR,external_inputs=F6_INPUTS,
               external_rows=5220,external_batches=parent["steps"],external_exposure=40*parent["steps"],
               event_samples=EVENT,aux_temporal_samples=PREFIX,torch=parent["torch"],braindecode=parent["braindecode"])
    results={}
    for kind in KINDS:
        p=HERE/"checkpoints"/f"{kind}_{tag}.pt"
        meta=json.loads(p.with_suffix(".json").read_text())
        expected=dict(fixed,kind=kind,checkpoint=sha(p),external_weight=0. if kind=="hard" else WEIGHT,
                      actual_aux=0 if kind=="hard" else parent["steps"])
        if (any(meta[k]!=v for k,v in expected.items()) or "H100" not in meta["device"] or
                (meta["state"]==parent["state"])!=(kind!="joint") or meta["state"]==meta["initial"]):
            raise ValueError("wrong all-source state, recipe or source population")
        if kind=="hard":
            if meta["checkpoint"]!=parent["checkpoint"]:
                raise ValueError("exact hard checkpoint bytes changed")
        else:
            probes=[meta[k] for k in ("external_body_gradient","external_head_gradient","external_norm_gradient",
                                       "external_final_loss","tail_drift")]
            if (not np.isfinite(probes).all() or min(probes[1:4])<=1e-8 or
                    (probes[0]<=1e-8 if kind=="joint" else probes[0]!=0) or not 0<=probes[-1]<=TOL or
                    not 0<meta["filter_radius"]<=EVENT-PREFIX or smoke and probes[3]>=log(5) or
                    not meta["native_norm_isolated"] or meta["auxiliary_initial"]==meta["auxiliary_final"]):
                raise ValueError("inactive or unisolated source motor objective")
        results[kind]=(p,meta)
    aux=[results[k][1] for k in ("detached","joint")]
    if (len({(m["external_tape"],m["external_scale"]) for _,m in results.values()})!=1 or
            aux[0]["auxiliary_initial"]!=aux[1]["auxiliary_initial"]):
        raise ValueError("external arms did not share source tape and initial auxiliary")
    return results


def run(seed,smoke):
    torch.set_num_threads(1)
    device=torch.cuda.get_device_name()
    if str(torch.__version__)!="2.14.1+cu130" or "RTX 2080 Ti" not in device:
        raise ValueError("requires the pinned native RTX worker")
    origin_report()
    x,rows,streams,people,_=scope(DATA)
    names=np.unique(streams)
    picked=[np.flatnonzero(streams==n)[:48] for n in (names[0],names[-1])]
    sample=[np.ascontiguousarray(x[i]) for i in picked]
    if any(a.shape!=(48,41,480) for a in sample):
        raise ValueError("real source reset probe is incomplete")
    out=HERE/("smoke_exports" if smoke else "exports")
    results={}
    for kind,(p,meta) in frozen_sources(seed,smoke,rows,streams,people).items():
        model,layer=frozen_member(p,meta)
        ref,clips=reference(model,layer,sample,meta["scale"])
        actor=export_actor(model,layer,meta["scale"],torch.from_numpy(sample[0][:1]))
        with torch.jit.optimized_execution(False):
            reset(actor)
        folder=out/kind
        folder.mkdir(exist_ok=True,parents=True)
        name=folder/f"raw_s1c_s{seed}.pt"
        partial=folder/f".{name.name}.pending"
        if partial.exists() or name.exists():
            raise FileExistsError(name)
        torch.jit.save(actor,str(partial))
        gpu=torch.jit.load(str(partial),map_location="cuda:0").eval()
        proof=check(gpu,sample,ref)
        partial.rename(name)
        results[kind]=dict(proof,path=str(name),sha256=sha(name),checkpoint=sha(p),state=meta["state"],source_receipt=sha(p.with_suffix(".json")),
                           bounded_active=clips,recipe=meta["external_tape"])
    record=dict(seed=seed,smoke=smoke,scherer_exposed=False,gold_opened=False,input=DATA_SHA,source_rows=row_sha(rows),
                probe_rows=[row_sha(i) for i in picked],actor=ACTOR,f16e_lock=F16E_LOCK,
                exporter=sha(HERE/"export.sha256"),torch=str(torch.__version__),device=device,results=results)
    path=out/f"s{seed}.json"
    with path.open("x") as handle:
        handle.write(json.dumps(record,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(f"PASS F32e native worker seed {seed} receipt {sha(path)}",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed",type=int,choices=(0,1))
    parser.add_argument("--smoke",action="store_true")
    options=parser.parse_args()
    run(options.seed,options.smoke)
