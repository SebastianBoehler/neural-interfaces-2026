"""Seal complete native tails with the exact F32 hard, detached and joint inputs."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from bridge_orders import write_npz,write_receipt
from bridge_owner import pin,sha,stamp
from complete_zero import FEAT,HASHES
from locked32 import ACTOR,ARMS,C20,C20_ARRAYS,C20_PUPILS,C20_RECEIPTS,F32,F32_BLIND,PUPILS
from outer32 import arms
from worker import run_tail

WORK=Path(__file__).resolve().parent
DELIVERY_HASH="6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f"


def inputs(order,people,source):
    pin(F32/"actor.sha256",ACTOR)
    pin(F32/"before_labels.sha256",F32_BLIND)
    parent={key:val for val,key in (line.split() for line in (F32/"before_labels.sha256").read_text().splitlines())}
    pin(C20/f"{order}.json",C20_RECEIPTS[order])
    inherited=json.loads((C20/f"{order}.json").read_text())["folds"]
    banks={arm:np.full((2,2655,3),np.nan) for arm in PUPILS}
    folded={}
    clips={arm:0 for arm in PUPILS}
    for person in sorted(set(people)):
        wanted=np.flatnonzero(people==person)
        for seed in (0,1):
            key=f"{person}_s{seed}.npz"
            path=WORK/"pupils"/key
            assert not path.stat().st_mode&0o222
            folded[key]=sha(path)
            with np.load(path,allow_pickle=False) as z, np.load(C20_PUPILS/"pupils"/key,allow_pickle=False) as hard:
                metadata=json.loads(str(z["metadata"]))
                expected=dict(person=person,seed=seed,source=source,actor=ACTOR,held_labels_opened=False,
                              data=HASHES[FEAT],deliveries=DELIVERY_HASH,f32_blind=F32_BLIND,
                              fold=parent["folds/"+key],c20=inherited[key])
                assert all(metadata[k]==v for k,v in expected.items())
                pin(F32/"folds"/key,expected["fold"])
                pin(C20_PUPILS/"pupils"/key,expected["c20"])
                np.testing.assert_array_equal(z["index"],wanted)
                np.testing.assert_array_equal(hard["index"],wanted)
                for arm,b in banks.items():
                    p=z[f"{arm}|{order}"]
                    proof=metadata["evidence"][arm]["orders"][order]
                    assert proof["active"]>=0 and proof["first"]>0 and 0<=proof["prefix"]<=2e-5
                    clips[arm]+=proof["active"]
                    assert p.shape==(len(wanted),3) and np.isnan(b[seed,wanted]).all()
                    if arm!="joint":
                        np.testing.assert_array_equal(p,hard[f"hard|{order}"])
                    b[seed,wanted]=p
    assert len(folded)==20 and len(parent)==20 and min(clips.values())>0
    for b in banks.values():
        assert np.isfinite(b).all() and np.min(b)>=0
        np.testing.assert_allclose(b.sum(2),1.,atol=1e-12,rtol=0)
    np.testing.assert_array_equal(banks["hard"],banks["detached"])
    assert not np.array_equal(banks["hard"],banks["joint"])
    return dict(identity=None,**{k:v.mean(0) for k,v in banks.items()}),folded


def main(order,mode,pupil_source):
    start=perf_counter()
    pin(WORK/"pupil.sha256",pupil_source)
    source=sha(WORK/"consumer.sha256")
    smoke=mode=="smoke"
    if not smoke:
        witness=json.loads((WORK/"outer_smoke.json").read_text())
        assert witness["passed"] and witness["source"]==source and witness["pupil_source"]==pupil_source
        assert not witness["held_labels_opened"]
    pin(FEAT,HASHES[FEAT])
    with np.load(FEAT,allow_pickle=False) as z:
        people=np.array([p.rsplit("/",1)[-1] for p in z["subject"]]); visits=z["session"]
    assert len(set(people))==10
    posteriors,folds=inputs(order,people,pupil_source)
    active=(people=="P01") if smoke else np.ones(len(people),dtype=bool)
    pin(C20/f"{order}.npz",C20_ARRAYS[order])
    with arms(order,posteriors,active) as ctx,np.load(C20/f"{order}.npz",allow_pickle=False) as old:
        count=ctx["count"]
        complete=tuple(count.SESS)
        rows={s:count.C.order_rows(order,s,count.PROTO) for s in complete}
        assert len(rows)==14
        p01=next(s for s,r in rows.items() if set(people[r])=={"P01"} and set(visits[r])=={"1"})
        latches=count.latch(order)
        peer=next(s for s in complete if s!=p01 and latches[s] is not None and latches[s]<120)
        prefixes=(p01,peer,p01)
        select=(p01,peer) if smoke else complete
        index=np.concatenate([rows[s] for s in select])
        assert len(np.unique(index))==len(index)
        if not smoke:
            np.testing.assert_array_equal(index,ctx["delivery"])
        np.testing.assert_array_equal(old["delivery"],ctx["delivery"])
        path,work={},{}
        for arm in ARMS:
            path[arm],work[arm]=run_tail(ctx,order,ctx["raw"][arm],select,prefixes)
        hard_rows=rows[p01] if smoke else index
        for stage in ("raw","filtered","terminal"):
            s=path["stock"][stage]
            np.testing.assert_array_equal(s[index],ctx["reference"][stage][index])
            np.testing.assert_array_equal(s[index],old[f"stock|{stage}"][index])
            np.testing.assert_array_equal(s[index],path["identity"][stage][index])
            np.testing.assert_array_equal(path["hard"][stage][hard_rows],old[f"hard|{stage}"][hard_rows])
            np.testing.assert_array_equal(path["hard"][stage][index],path["detached"][stage][index])
            if smoke:
                for arm in PUPILS:
                    np.testing.assert_array_equal(path[arm][stage][rows[peer]],s[rows[peer]])
        fit_hash={k:{str(s):v["csp"][str(s)]["target_sha"] for s in select} for k,v in work.items()}
        assert fit_hash["stock"]==fit_hash["identity"] and fit_hash["hard"]==fit_hash["detached"]
        assert fit_hash["joint"]!=fit_hash["hard"] and fit_hash["joint"]!=fit_hash["stock"]
        maximum={k:float(np.abs(path[k]["terminal"][index]-path["stock"]["terminal"][index]).max()) for k in PUPILS}
        assert min(maximum.values())>1e-4
        assert np.abs(path["joint"]["terminal"][index]-path["hard"]["terminal"][index]).max()>1e-4
        for w in work.values():
            assert w["count_effect"]>0
            assert (w["csp_refits"],w["csp_fusions"])==((18,sum(len(rows[s])-90 for s in select)) if smoke else (126,1395))
        changed={k:int((path[k]["terminal"][index].argmax(1)!=path["stock"]["terminal"][index].argmax(1)).sum()) for k in PUPILS}
        proof=dict(passed=True,scope=mode,order=order,source=source,pupil_source=pupil_source,parent=ctx["parent"],
                   archive_zero=ctx["zero"],warm_steps=ctx["warm"],rows=len(index),sessions=len(select),folds=folds,
                   prefixes=[str(s) for s in prefixes],held_labels_opened=False,exact_stock=True,exact_identity=True,
                   equal_hard_detach=True,outer=ctx["proof"],private_past_csp=True,work=work,
                   terminal_hash={k:stamp(v["terminal"][index]) for k,v in path.items()},
                   matched_terminal_max=maximum,changed=changed,slot_mass=5/6,seconds=perf_counter()-start)
        if not smoke:
            bank=dict(index=np.arange(len(index)),delivery=index,**{f"{a}|{k}":v for a in path for k,v in path[a].items()})
            proof["output"]=write_npz(WORK/f"{order}.npz",bank)
        file=WORK/("outer_smoke.json" if smoke else order+".json")
        write_receipt(file,**proof)
    print("PASS F32c outer",order,mode,sha(file),changed,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("order",choices=("within0","full0"))
    p.add_argument("mode",choices=("smoke","complete"))
    p.add_argument("pupil_source")
    a=p.parse_args()
    assert a.mode!="smoke" or a.order=="full0"
    main(a.order,a.mode,a.pupil_source)
