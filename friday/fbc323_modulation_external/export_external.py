"""Qualify F25e's untouched source-trained models in the CPU worker image."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from anchors import DATA_SHA,ORIGIN_SHA,RUNTIME,row_sha,sha
from export_checks import check,reference,reset
from held_modulation import frozen_member
from incremental_model import export_actor
from modulation import ACTIVE,KINDS,MAX_RATIO,MAX_RESERVED,architecture
from replay_f13 import DATA,origin_report
from source_scope import F16E_LOCK,authenticated,scope


HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/"ni26_friday_s1c_motor_external_cpu_20261119/anchors_f16e"
ACTOR="b9d64a585b6f75a806a0d27c8b04fd32d41c8baf565a46ac07e7835e2c673e4b"
F25="589b9a212cafdb0367f9f34bb840db5ff95c98ccfeea3bf77e268aa3ddb2e95c"
GPU="NVIDIA A100-PCIE-40GB"


def authenticated_group(seed,smoke,rows,streams,people):
    if sha(PARENT/"before_labels.json")!=F16E_LOCK or sha(HERE/"actor.sha256")!=ACTOR:
        raise ValueError("locked all-source baseline or A100 source actor differs")
    parent,batches=authenticated(PARENT,PARENT,seed,smoke,rows,streams,people)
    flat=[r for ep in batches for r in ep]
    tag=f"{'smoke_' if smoke else ''}{seed}_all_source"
    fixed=dict(held="all_source",seed=seed,epochs=parent["epochs"],steps=parent["steps"],dataset=DATA_SHA,origin=ORIGIN_SHA,
               source_rows=row_sha(rows),source_people=people,source_count=2655,source_sessions=14,batches=parent["batches"],
               runtime_reference=RUNTIME,weights_origin=parent["checkpoint"],h100_state=parent["state"],initial=parent["initial"],
               actor=ACTOR,f25=F25,source_only_labels=True,held_labels_opened=False,scherer_exposed=False,zero_identity=True,
               rng_isolated=True,q_gradients_nonzero=0,diagnostic_rows=row_sha(rows[flat[0][:32]]),device=GPU,
               torch="2.14.1+cu126",braindecode="1.8.1")
    group={}
    for kind in KINDS:
        path=HERE/"checkpoints"/f"{kind}_{tag}.pt"
        m=json.loads(path.with_suffix(".json").read_text())
        exact=dict(fixed,kind=kind,checkpoint=sha(path),**architecture(kind))
        if (any(m[k]!=v for k,v in exact.items()) or
                not np.isclose(m["scale"],parent["scale"],atol=0,rtol=2e-6) or
                not 0<m["warm_ms"]<=MAX_RATIO*m["pinned_hard_ms"] or not 0<m["peak_reserved"]<=MAX_RESERVED or
                m["state"]==m["initial"]):
            raise ValueError("invalid matched all-source fit receipt")
        if kind=="hard":
            if (m["state"]!=m["same_device_hard_state"] or m["checkpoint"]!=m["same_device_hard_checkpoint"] or
                    m["warm_ms"]!=m["pinned_hard_ms"]):
                raise ValueError("independent same-device hard identity differs")
        else:
            checks=[m[k] for k in ("band_grad","tap_grad","identity_delta","dilation_delta","cross_band","edge_tap")]
            if not np.isfinite(checks).all() or min(checks)<=ACTIVE or not .75<=m["gain_min"]<m["gain_max"]<=1.25:
                raise ValueError("trained F25 gate was inactive")
        group[kind]=path,m
    keys=("same_device_hard_state","same_device_hard_checkpoint","scale","cudnn","cudnn_tf32","matmul_tf32","pinned_hard_ms")
    if len({m["state"] for _,m in group.values()})!=3 or any(len({m[k] for _,m in group.values()})!=1 for k in keys):
        raise ValueError("paired gate state or device recipes differ")
    return group


def run(seed,smoke):
    if str(torch.__version__)!="2.14.1+cu130":
        raise ValueError("wrong original CPU worker image")
    torch.set_num_threads(1)
    origin_report()
    x,rows,streams,people,_=scope(DATA)
    names=np.unique(streams)
    picked=[np.flatnonzero(streams==n)[:48] for n in (names[0],names[-1])]
    sample=[np.ascontiguousarray(x[i]) for i in picked]
    if any(a.shape!=(48,41,480) for a in sample):
        raise ValueError("real source-only export replay is incomplete")
    out=HERE/("smoke_exports" if smoke else "exports")
    results={}
    for kind,(path,meta) in authenticated_group(seed,smoke,rows,streams,people).items():
        model,layer=frozen_member(path,meta)
        truth,active=reference(model,layer,sample,meta["scale"])
        if active<=0:
            raise ValueError("real source fixture did not activate bounded normalization")
        actor=export_actor(model,layer,meta["scale"],torch.from_numpy(sample[0][:1]))
        with torch.jit.optimized_execution(False):
            reset(actor)
        folder=out/kind
        folder.mkdir(parents=True,exist_ok=True)
        target=folder/f"raw_s1c_s{seed}.pt"
        pending=folder/f".{target.name}.pending"
        if target.exists() or pending.exists():
            raise FileExistsError(target)
        torch.jit.save(actor,str(pending))
        cpu=torch.jit.load(str(pending),map_location="cpu").eval()
        proof=check(cpu,sample,truth)
        pending.rename(target)
        result=dict(proof,path=str(target),sha256=sha(target),state=meta["state"],checkpoint=sha(path),
                    source_receipt=sha(path.with_suffix(".json")),bounded_active=active,dilations=meta["dilations"])
        results[kind]=result
    receipt=dict(seed=seed,smoke=smoke,device="cpu",fit_device=GPU,torch=str(torch.__version__),actor=ACTOR,
                 exporter=sha(HERE/"export.sha256"),f16e_lock=F16E_LOCK,f25=F25,input=DATA_SHA,source_rows=row_sha(rows),
                 probe_rows=[row_sha(i) for i in picked],scherer_exposed=False,gold_opened=False,results=results)
    path=out/f"s{seed}.json"
    with path.open("x") as handle:
        handle.write(json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(f"PASS F25e CPU seed {seed} receipt {sha(path)}",flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("seed",type=int,choices=(0,1))
    cli.add_argument("--smoke",action="store_true")
    args=cli.parse_args()
    run(args.seed,args.smoke)
