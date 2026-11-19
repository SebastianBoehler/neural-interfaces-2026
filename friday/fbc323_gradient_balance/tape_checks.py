"""Recompute F35 source-only weight and diagnostic algebra on the frozen fit tape."""
import json

import numpy as np

from anchors import DATA_SHA,ORIGIN_SHA,row_sha,sha,split
from contributions import CAP,HERE,KINDS,PARAMETERS,eligibility,imbalance,transfer
from group import ACTOR as F33A
from group import paired as anchor
from recipe import BD,CPU_CUDA,GPU,TORCH,canary,historical,tag


ACTOR="b153df05e923c0b40795adaed413d8d024087e49e4b2be51445d81b6180db42e"
F33=HERE.parent/"ni26_friday_s1c_covariance_a100_whole_20261119"
F33_BLIND="f996e3a55d376f6856bbd07425632033924a175636acc21b332d1a4d3f4f0525"


def read_tape(path,expected,eligible):
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(str(z["metadata"]))
        values,mask,applied=(z[k] for k in ("trace","eligible","applied"))
    active=values[eligible]
    if (any(meta[k]!=v for k,v in expected.items()) or not np.array_equal(mask,eligible) or
            values.shape!=(len(eligible),4) or applied.shape!=eligible.shape or
            values.dtype!=np.float64 or applied.dtype!=np.float64 or mask.dtype!=bool or
            not np.isfinite(values).all() or not np.isfinite(applied).all() or
            np.any(values[~eligible]) or np.any(applied[~eligible]) or
            np.any(active[:,:2]<=CAP) or np.any(active[:,2:]<0) or np.any(active[:,2:].sum(1)<=0)):
        raise ValueError("source trajectory/tape rows, group scope or numerical activity differs")
    return values,applied,meta


def paired(held,seed,smoke,subject,visit):
    if sha(HERE/"actor.sha256")!=ACTOR or sha(F33/"before_labels.sha256")!=F33_BLIND:
        raise ValueError("F35 source or exact A100 held ancestry differs")
    rows,streams,people=split(subject,visit,held)
    ref,batches,_=historical(held,seed,smoke,rows,streams,people)
    native=anchor(held,seed,smoke,subject,visit)["hard"][1]
    env=canary()
    name=tag(held,seed,smoke)
    flat=[r for epoch in batches for r in epoch]
    eligible=eligibility(flat,streams)
    fixed=dict(held=held,seed=seed,epochs=ref["epochs"],scale=native["scale"],initial=ref["initial"],data=DATA_SHA,
               origin=ORIGIN_SHA,source_rows=row_sha(rows),source_people=people,batches=ref["batches"],
               held_labels_opened=False,actor=ACTOR,f33a=F33A,weights_origin=native["checkpoint"],device=GPU,
               torch=TORCH,braindecode=BD,parameters=PARAMETERS,q_gradients_nonzero=0,updates_order_exact=True,
               **{k:env[k] for k in CPU_CUDA})
    tape=HERE/"checkpoints"/f"tape_{name}.npz"
    lock=sha(tape)
    t,used,meta=read_tape(tape,dict(fixed,hard_state=native["state"]),eligible)
    zero=np.zeros_like(used)
    if not np.array_equal(used,transfer(t,eligible)) or not 0<=meta["hard_gradient_error"]<=2e-6:
        raise ValueError("derived frozen mass transfer or analytical gradient proof differs")
    before=imbalance(t,eligible,zero)[0]
    balanced=imbalance(t,eligible,used)[0]
    mirrored=imbalance(t,eligible,-used)[0]
    relative=1-balanced/before
    anchor_values=dict(anchor_original=before,anchor_balanced=balanced,anchor_mirrored=mirrored)
    if any(meta[k]!=v for k,v in anchor_values.items()) or relative<.10 or not mirrored>before:
        raise ValueError("source tape did not change matched anchor imbalance")
    result={}
    for kind,sign in zip(KINDS,(0,1,-1),strict=True):
        path=HERE/"checkpoints"/f"{kind}_{name}.pt"
        m=json.loads(path.with_suffix(".json").read_text())
        expected=dict(fixed,kind=kind,checkpoint=sha(path),steps=ref["steps"],tape=lock,eligible=int(eligible.sum()))
        if (any(m[k]!=v for k,v in expected.items()) or m["state"]==m["initial"] or
                not np.isfinite([m[k] for k in ("first_loss","last_loss","max_gradient")]).all() or m["max_gradient"]<=0):
            raise ValueError(f"bad fit identity or exact ordered exposure for {kind}")
        if kind=="hard":
            identity=dict(anchor_values,hard_gradient_error=meta["hard_gradient_error"],anchor_relative_reduction=relative,
                          checkpoint=native["checkpoint"],state=native["state"])
            if any(m[k]!=v for k,v in identity.items()):
                raise ValueError("actual hard A100 state or trajectory differs")
        else:
            path_trace=HERE/"checkpoints"/f"{kind}_trajectory_{name}.npz"
            trace,d,_=read_tape(path_trace,dict(fixed,kind=kind,state=m["state"],tape=lock),eligible)
            a0,r0=imbalance(trace,eligible,zero)
            a1,r1=imbalance(trace,eligible,d)
            proof=dict(trajectory=sha(path_trace),sign=sign,actual_original=a0,actual_applied=a1,
                       actual_relative_original=r0,actual_relative_applied=r1)
            if not np.array_equal(d,sign*used) or any(m[k]!=v for k,v in proof.items()):
                raise ValueError("actual applied mass or executed imbalance differs")
        result[kind]=path,m
    if len({m["state"] for _,m in result.values()})!=len(KINDS):
        raise ValueError("matched source states are duplicated")
    return result
