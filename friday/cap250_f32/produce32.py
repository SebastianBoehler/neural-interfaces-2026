"""Recompute native bounded posterior folds and match hard to its exact ancestor."""

import argparse
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from anchors import sha
from bounded_bn import normalize
from encoder312 import fingerprint
from held import frozen_member
from held_distill import raw_probability
from held_joint import external_metadata, paired
from locked32 import ACTOR,C20,C20_PUPILS,C20_PUPIL_SOURCE,C20_RECEIPTS,F32,F32_BLIND,F32_RESULT,F32_SCORER,ORDERS,PUPILS
from pupils import do_order
from replay_f13 import DATA_SHA,ORDER_SHA,ORDERS as ORIGINAL_ORDERS,origin_report,raw_inputs


WORK=Path(__file__).resolve().parent


@torch.no_grad()
def first_windows(model,layer,x,stream,rows,delivery,scale):
    original=layer.forward
    first=[delivery[stream[delivery]==s][0] for s in dict.fromkeys(stream[rows])]
    counts=[]
    for row in first:
        current=x[row:row+1]/scale
        q=raw_probability(model,current)
        try:
            layer.forward=MethodType(lambda this,value: normalize(this,value,counts),layer)
            np.testing.assert_array_equal(q,raw_probability(model,current))
        finally:
            layer.forward=original
    assert len(counts)==len(first)
    return len(first)


def run(held,seed,scope):
    torch.set_num_threads(2)
    origin_report()
    assert sha(F32/"actor.sha256")==ACTOR and sha(F32/"score.sha256")==F32_SCORER
    assert sha(F32/"before_labels.sha256")==F32_BLIND and sha(F32/"whole_results.json")==F32_RESULT
    assert sha(C20/"full0.json")==C20_RECEIPTS["full0"] and sha(C20_PUPILS/"source.sha256")==C20_PUPIL_SOURCE
    x,subject,people,stream,perms=raw_inputs()
    visit=np.array([v.rsplit("|",1)[-1] for v in stream])
    rows=np.flatnonzero(people==held)
    name=f"{held}_s{seed}.npz"
    path=F32/"folds"/name
    prior=C20_PUPILS/"pupils"/name
    manifest={p:h for h,p in (line.split() for line in (F32/"before_labels.sha256").read_text().splitlines())}
    assert len(manifest)==20 and sha(path)==manifest["folds/"+name]==path.with_suffix(".sha256").read_text().strip()
    assert sha(prior)==json.loads((C20/"full0.json").read_text())["folds"][name]
    states=paired(held,seed,False,subject,visit,external_metadata())
    results={}
    proof={}
    with np.load(path,allow_pickle=False) as sealed,np.load(prior,allow_pickle=False) as old:
        meta=json.loads(str(sealed["metadata"]))
        expected=dict(held=held,seed=seed,actor=ACTOR,scorer=F32_SCORER,data=DATA_SHA,
                      order_sha=ORDER_SHA,orders=list(ORIGINAL_ORDERS),held_labels_opened=False)
        assert all(meta[k]==v for k,v in expected.items()) and set(states)==set(PUPILS)
        np.testing.assert_array_equal(sealed["index"],rows)
        np.testing.assert_array_equal(old["index"],rows)
        for arm in PUPILS:
            checkpoint,fit=states[arm]
            evidence=meta["evidence"][arm]
            assert evidence["checkpoint"]==fit["checkpoint"] and evidence["state"]==fit["state"]
            assert evidence["receipt"]==sha(checkpoint.with_suffix(".json")) and evidence["tape"]==fit["external_tape"]
            assert set(fit["source_people"])==set(subject[people!=held]) and fit["source_labels_only"]
            model,layer=frozen_member(checkpoint,fit)
            model.skip=False
            source={}
            for order in ORDERS:
                p,delta,clips=do_order(model,layer,x,stream,rows,perms[order],fit["scale"])
                first=first_windows(model,layer,x,stream,rows,perms[order],fit["scale"])
                np.testing.assert_array_equal(p.argmax(1),sealed[f"{arm}_bounded"][ORIGINAL_ORDERS.index(order)])
                if arm in ("hard","detached"):
                    np.testing.assert_array_equal(p,old[f"hard|{order}"])
                assert clips>=0 and first>0 and 0<=delta<=2e-5
                results[f"{arm}|{order}"]=p
                source[order]=dict(prefix=delta,active=clips,first=first)
            assert fingerprint(model)==fit["state"]
            proof[arm]=dict(checkpoint=fit["checkpoint"],state=fit["state"],receipt=evidence["receipt"],orders=source)
    assert proof["hard"]["state"]==proof["detached"]["state"]!=proof["joint"]["state"]
    for order in ORDERS:
        assert not np.array_equal(results[f"joint|{order}"],results[f"hard|{order}"])
    output=WORK/scope/name
    metadata=dict(person=held,seed=seed,actor=ACTOR,f32_blind=F32_BLIND,fold=sha(path),c20=sha(prior),
                  source=sha(WORK/"pupil.sha256"),data=DATA_SHA,deliveries=ORDER_SHA,
                  evidence=proof,held_labels_opened=False)
    with output.open("xb") as f:
        np.savez_compressed(f,index=rows,metadata=json.dumps(metadata,sort_keys=True,allow_nan=False),**results)
    output.chmod(0o444)
    print("PASS F32c pupils",held,seed,scope,sha(output),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("held")
    p.add_argument("seed",type=int,choices=(0,1))
    p.add_argument("scope",choices=("smoke","pupils"))
    a=p.parse_args()
    run(a.held,a.seed,a.scope)
