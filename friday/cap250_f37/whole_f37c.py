"""Replay alpha and beta inside the matched F32c H100 native cap250 owner."""

import argparse
import json
from time import perf_counter

import numpy as np

from bridge_orders import write_npz,write_receipt
from bridge_owner import pin,sha,stamp
from complete_zero import FEAT,HASHES
from inputs37 import read_pupils
from outer37 import arms
from pins_f37c import ARMS,C20,C20_ARRAYS,F32C,F32_ARRAYS,F32_RECEIPTS,PUPILS,WORK
from worker import run_tail

def same_hard(order, paths, stock_rows, hard_rows):
    for parent,pins in ((C20,C20_ARRAYS),(F32C,F32_ARRAYS)):
        file=parent/f"{order}.npz"
        pin(file,pins[order])
        with np.load(file,allow_pickle=False) as prior:
            for arm,indices in (("stock",stock_rows),("hard",hard_rows)):
                for stage,values in paths[arm].items():
                    np.testing.assert_array_equal(values[indices],prior[f"{arm}|{stage}"][indices])


def main(order,mode):
    start=perf_counter()
    source={k:sha(WORK/f"{k}.sha256") for k in ("pupil","whole")}
    smoke=mode=="smoke"
    if not smoke:
        witness=json.loads((WORK/"outer_smoke.json").read_text())
        assert witness["passed"] and witness["source"]==source
        assert not witness["held_labels_opened"]
    pin(FEAT,HASHES[FEAT])
    with np.load(FEAT,allow_pickle=False) as z:
        people=np.array([p.rsplit("/",1)[-1] for p in z["subject"]]); visits=z["session"]
    assert len(set(people))==10
    posteriors,folds,fits,clips=read_pupils(order,people,visits,source["pupil"])
    if not smoke:
        assert witness["folds"]==folds and witness["fit_receipts"]==fits
    active=(people=="P01") if smoke else np.ones(len(people),dtype=bool)
    pin(C20/f"{order}.npz",C20_ARRAYS[order])
    proof_path=F32C/f"{order}.json"
    pin(proof_path,F32_RECEIPTS[order])
    ancestor=json.loads(proof_path.read_text())
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
        same_hard(order,path,index,hard_rows)
        assert ctx["warm"]==ancestor["warm_steps"]==9600
        assert all(ctx["proof"][a][k]==ancestor["outer"]["identity"][k]
                   for a in ARMS[1:] for k in ("shadow","gate","refit_states"))
        if not smoke:
            assert ctx["proof"]["hard"]==ancestor["outer"]["hard"]
        for stage in ("raw","filtered","terminal"):
            s=path["stock"][stage]
            np.testing.assert_array_equal(s[index],ctx["reference"][stage][index])
            np.testing.assert_array_equal(s[index],old[f"stock|{stage}"][index])
            np.testing.assert_array_equal(s[index],path["identity"][stage][index])
            if smoke:
                for arm in PUPILS:
                    np.testing.assert_array_equal(path[arm][stage][rows[peer]],s[rows[peer]])
        fit_hash={k:{str(s):v["csp"][str(s)]["target_sha"] for s in select} for k,v in work.items()}
        assert fit_hash["stock"]==fit_hash["identity"]
        assert all(fit_hash[a]!=fit_hash["stock"] for a in PUPILS)
        maximum={k:float(np.abs(path[k]["terminal"][index]-path["stock"]["terminal"][index]).max()) for k in PUPILS}
        assert min(maximum.values())>1e-4
        assert all(np.abs(path[a]["terminal"][index]-path[b]["terminal"][index]).max()>1e-4
                   for a,b in (("alpha","hard"),("beta","hard"),("beta","alpha")))
        for w in work.values():
            assert w["count_effect"]>0
            assert (w["csp_refits"],w["csp_fusions"])==((18,sum(len(rows[s])-90 for s in select)) if smoke else (126,1395))
        changed={k:int((path[k]["terminal"][index].argmax(1)!=path["stock"]["terminal"][index].argmax(1)).sum()) for k in PUPILS}
        proof=dict(passed=True,scope=mode,order=order,source=source,parent=ctx["parent"],
                   archive_zero=ctx["zero"],warm_steps=ctx["warm"],rows=len(index),sessions=len(select),folds=folds,
                   prefixes=[str(s) for s in prefixes],held_labels_opened=False,exact_stock=True,exact_identity=True,
                   exact_h100_hard=True,outer=ctx["proof"],private_past_csp=True,work=work,
                   fit_receipts=fits,bounded_clips=clips,
                   terminal_hash={k:stamp(v["terminal"][index]) for k,v in path.items()},
                   matched_terminal_max=maximum,changed=changed,slot_mass=5/6,seconds=perf_counter()-start)
        if not smoke:
            bank=dict(index=np.arange(len(index)),delivery=index,**{f"{a}|{k}":v for a in path for k,v in path[a].items()})
            proof["output"]=write_npz(WORK/f"{order}.npz",bank)
        file=WORK/("outer_smoke.json" if smoke else order+".json")
        write_receipt(file,**proof)
    print("PASS F37c outer",order,mode,sha(file),changed,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("order",choices=("within0","full0"))
    p.add_argument("mode",choices=("smoke","complete"))
    a=p.parse_args()
    assert a.mode!="smoke" or a.order=="full0"
    main(a.order,a.mode)
