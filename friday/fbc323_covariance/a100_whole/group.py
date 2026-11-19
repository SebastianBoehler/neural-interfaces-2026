"""Independently authenticate the A100 source folds without using H100 weights."""
import json

import numpy as np

from anchors import DATA_SHA,ORIGIN_SHA,row_sha,sha,split
from covariance import BASE,EXTRAS,KINDS,MAX_RATIO,MAX_RESERVED,order
from recipe import BD,CANARY_SHA,CPU_CUDA,F33,GPU,HERE,TORCH,canary,historical,tag


ACTOR="94598f4eb485b20ea40c0d7c13f4497ed07d04e9e9c062c72953e97460b782bc"


def checked_receipt(path,expected,scale):
    m=json.loads(path.with_suffix(".json").read_text())
    if (any(m[k]!=v for k,v in expected.items()) or m["checkpoint"]!=sha(path) or
            not np.isclose(m["scale"],scale,atol=0,rtol=2e-6) or not 0<m["scale"] or
            not np.isfinite([m[k] for k in ("first_loss","last_loss","max_gradient")]).all() or
            m["max_gradient"]<=0 or m["state"]==m["initial"]):
        raise ValueError(f"source recipe or authenticated {path.name} receipt differs")
    return m


def paired(held,seed,smoke,subject,visit):
    if sha(HERE/"actor.sha256")!=ACTOR:
        raise ValueError("F33a independent source actor differs")
    observed=canary()
    source,streams,people=split(subject,visit,held)
    ref,batches,receipt=historical(held,seed,smoke,source,streams,people)
    flat=[rows for ep in batches for rows in ep]
    fixed=dict(held=held,seed=seed,epochs=ref["epochs"],initial=ref["initial"],data=DATA_SHA,origin=ORIGIN_SHA,
               source_rows=row_sha(source),source_people=people,batches=ref["batches"],historical_metadata=receipt,
               h100_state=ref["state"],h100_weights_loaded=False,held_labels_opened=False,actor=ACTOR,
               f33=F33,canary=CANARY_SHA,device=GPU,torch=TORCH,braindecode=BD,steps=ref["steps"],
               **{k:observed[k] for k in CPU_CUDA})
    name=tag(held,seed,smoke)
    folder=HERE/"checkpoints"
    path=folder/f"bare_{name}.pt"
    bare=checked_receipt(path,dict(fixed,kind="hard",role="bare",parameters=BASE),ref["scale"])
    if smoke and (bare["state"]!=observed["a100_state"] or bare["scale"]!=observed["scale"]):
        raise ValueError("bare A100 source smoke differs from the registered same-device canary")
    result={}
    matched=dict(fixed,role="matched",scale=bare["scale"],bare_checkpoint=bare["checkpoint"],bare_state=bare["state"],
                 train_rows_seen=sum(map(len,flat)),diagnostic_rows=row_sha(source[flat[0][:32]]),
                 rng_isolated=True,zero_identity=True,q_gradients_nonzero=0)
    for kind in KINDS:
        checkpoint=folder/f"{kind}_{name}.pt"
        m=checked_receipt(checkpoint,dict(matched,kind=kind,parameters=BASE+EXTRAS*(kind!="hard")),bare["scale"])
        if not 0<m["peak_reserved"]<=MAX_RESERVED:
            raise ValueError("invalid A100 memory evidence")
        if kind=="hard":
            if m["state"]!=bare["state"] or m["checkpoint"]!=bare["checkpoint"]:
                raise ValueError("A100 hard differs from independent bare native actor")
        else:
            deltas=[m[k] for k in ("left_gradient","right_gradient","relation_gradient","logit_delta","residual")]
            if (not np.isfinite(deltas).all() or min(deltas[:3])<=1e-8 or min(deltas[3:])<=1e-4 or
                    not 0<=m["energy_error"]<=1e-8 or not 0<=m["gain_error"]<=2e-6 or
                    not .05<m["feature_delta"]<=2.000004 or m["permutation"]!=row_sha(order(kind)) or
                    m["initial_basis"]==m["final_basis"]):
                raise ValueError("matched spatial alignment or invariance evidence differs")
        result[kind]=checkpoint,m
    members=[m for _,m in result.values()]
    if (len({m["state"] for m in members})!=3 or
            result["aligned"][1]["initial_basis"]!=result["permuted"][1]["initial_basis"]):
        raise ValueError("paired fitted fingerprint or initial basis differs")
    if smoke:
        hard_ms=result["hard"][1]["warm_ms"]
        if any(m["hard_ms"]!=hard_ms or not 0<m["warm_ms"]<=hard_ms*MAX_RATIO for m in members):
            raise ValueError("A100 paired warm timing evidence differs")
    result["bare"]=path,bare
    return result
