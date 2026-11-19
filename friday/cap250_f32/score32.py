"""Score the complete corrected paths only after independent blind authentication."""

import json
from pathlib import Path

import numpy as np

from bridge_orders import write_receipt
from bridge_owner import ROOT,pin,sha,stamp,start_module
from complete_score import METRIC,METRIC_HASH
from complete_zero import FEAT,HASHES
from locked32 import ARMS,C20,C20_ARRAYS,C20_SCORE,ORDERS,PUPILS
from member_inputs import ORDERS as PINNED_ORDERS,pinned_delivery


WORK=Path(__file__).resolve().parent
SCORED=("stock",*PUPILS)
PAIRS=(("joint","stock"),("joint","hard"),("joint","detached"),
       ("hard","detached"),("hard","stock"),("detached","stock"))


def authenticate(order,smoke):
    path,bank=WORK/f"{order}.json",WORK/f"{order}.npz"
    receipt=json.loads(path.read_text())
    assert receipt["passed"] and receipt["scope"]=="complete" and receipt["order"]==order
    assert not receipt["held_labels_opened"] and (receipt["rows"],receipt["sessions"])==(2655,14)
    assert receipt["exact_stock"] and receipt["exact_identity"] and receipt["equal_hard_detach"]
    assert receipt["private_past_csp"] and receipt["slot_mass"]==5/6 and receipt["warm_steps"]==9600
    assert receipt["source"]==smoke["source"]==sha(WORK/"consumer.sha256")
    assert receipt["pupil_source"]==smoke["pupil_source"]==sha(WORK/"pupil.sha256")
    assert receipt["folds"]==smoke["folds"] and len(receipt["folds"])==20
    assert all(sha(WORK/"pupils"/k)==v for k,v in receipt["folds"].items())
    assert receipt["prefixes"][0]==receipt["prefixes"][2]!=receipt["prefixes"][1]
    assert receipt["output"]["path"]==str(bank) and not (path.stat().st_mode|bank.stat().st_mode)&0o222
    for a in ARMS:
        w=receipt["work"][a]
        assert (w["csp_refits"],w["csp_fusions"])==(126,1395)
    outer=receipt["outer"]
    assert outer["identity"]["slot_movement"]==0. and outer["hard"]==outer["detached"]
    assert min(outer[a]["slot_movement"] for a in PUPILS)>.01
    for k in PUPILS:
        assert all(outer[k][stage]==outer["identity"][stage] for stage in ("shadow","gate","refit_states"))
    pin(bank,receipt["output"]["sha256"])
    prior=C20/f"{order}.npz"
    pin(prior,C20_ARRAYS[order])
    wanted=pinned_delivery(ROOT/"runs/shuffle_perms.npz")[PINNED_ORDERS.index(order)]
    with np.load(bank,allow_pickle=False) as here,np.load(prior,allow_pickle=False) as old:
        np.testing.assert_array_equal(here["index"],np.arange(2655))
        np.testing.assert_array_equal(here["delivery"],wanted)
        np.testing.assert_array_equal(old["delivery"],wanted)
        for stage in ("raw","filtered","terminal"):
            for arm in ("stock","hard"):
                np.testing.assert_array_equal(here[f"{arm}|{stage}"],old[f"{arm}|{stage}"])
            np.testing.assert_array_equal(here[f"stock|{stage}"],here[f"identity|{stage}"])
            np.testing.assert_array_equal(here[f"hard|{stage}"],here[f"detached|{stage}"])
        for a in ARMS:
            assert stamp(here[f"{a}|terminal"][wanted])==receipt["terminal_hash"][a]
        pred=np.stack([here[f"{a}|terminal"].argmax(1) for a in SCORED],axis=1)
    assert np.isin(pred,(0,1,2)).all()
    return pred,receipt,sha(path)


def metrics(pred,y,subject,groups,metric):
    people=sorted(set(subject))
    sums,weights=[],[]
    for name in people:
        m=subject==name
        cells=np.column_stack([metric.group_scores(metric.balanced_accuracy,y[m],v[m],groups[m]) for v in pred.T])
        sums.append(cells.sum(0)); weights.append(len(cells))
    s,w=np.asarray(sums),np.asarray(weights)
    assert len(people)==10 and w.sum()==28
    point=100*s.sum(0)/w.sum()
    draw=np.random.default_rng(460).integers(0,10,(4000,10))
    boot=100*s[draw].sum(1)/w[draw].sum(1)[:,None]
    contrasts={}
    for a,b in PAIRS:
        i,j=SCORED.index(a),SCORED.index(b)
        lo,hi=np.percentile(boot[:,i]-boot[:,j],(10,90)).tolist()
        delta=float(point[i]-point[j])
        v="exact" if a=="hard" and b=="detached" else "positive" if lo>0 else "refuted" if hi<=0 else "undecided"
        if v=="exact":
            assert lo==delta==hi==0.
        contrasts[f"{a}_minus_{b}"]=dict(pp=delta,paired80=[lo,hi],verdict=v)
    def changes(a,b,mask):
        p,q=(pred[:,SCORED.index(k)] for k in (a,b))
        return dict(switched=int(((p!=q)&mask).sum()),rescues=int(((p==y)&(q!=y)&mask).sum()),
                    losses=int(((p!=y)&(q==y)&mask).sum()))
    entities={"all":np.ones(len(y),dtype=bool),**{str(c):y==c for c in range(3)},**{p:subject==p for p in people}}
    movement={f"{a}_minus_{b}":{k:changes(a,b,m) for k,m in entities.items()} for a,b in PAIRS}
    return dict(absolute_pct=dict(zip(SCORED,point.tolist(),strict=True)),contrast=contrasts,movement=movement,
                people_pct={p:dict(zip(SCORED,(100*s[i]/w[i]).tolist(),strict=True)) for i,p in enumerate(people)})


def main():
    smoke=json.loads((WORK/"outer_smoke.json").read_text())
    assert smoke["passed"] and smoke["scope"]=="smoke" and smoke["order"]=="full0" and smoke["sessions"]==2
    assert not smoke["held_labels_opened"] and smoke["equal_hard_detach"]
    inputs={order:authenticate(order,smoke) for order in ORDERS}
    lock=WORK/"before_labels.json"
    write_receipt(lock,held_labels_opened=False,smoke=sha(WORK/"outer_smoke.json"),
                  orders={k:dict(proof=proof,bank=r["output"],predictions=stamp(p)) for k,(p,r,proof) in inputs.items()})
    pin(C20/"score.json",C20_SCORE)
    reference=json.loads((C20/"score.json").read_text())["orders"]
    pin(FEAT,HASHES[FEAT]); pin(METRIC,METRIC_HASH)
    metric=start_module("f32c_official",METRIC,METRIC.read_text())
    with np.load(FEAT,allow_pickle=False) as z:
        y,subject,visit,interface=(z[k] for k in ("y","subject","session","interface"))
    groups=np.stack((subject,visit,interface),axis=1)
    reports={order:metrics(p,y,subject,groups,metric) for order,(p,*_) in inputs.items()}
    for order,result in reports.items():
        for a in ("stock","hard"):
            assert abs(result["absolute_pct"][a]-reference[order]["absolute_pct"][a])<=1e-8
    advances=all(r["contrast"][f"joint_minus_{c}"]["verdict"]=="positive"
                 for r in reports.values() for c in ("stock","hard","detached"))
    path=WORK/"score.json"
    write_receipt(path,scope="FULL-SYSTEM DEV corrected PROTEUS held-new, reused ten people",
                  denominator="28 official cells per delivery",before_labels=sha(lock),
                  reports=reports,advance=advances,bootstrap=dict(people=4000,seed=460,interval=.80))
    print("PASS F32c score",sha(path),advances,{k:v["contrast"] for k,v in reports.items()},flush=True)


if __name__=="__main__":
    main()
