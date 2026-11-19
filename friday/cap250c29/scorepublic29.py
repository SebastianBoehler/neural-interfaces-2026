"""Independently authenticate and score both corrected full-system instruments."""
import json
from pathlib import Path

import numpy as np

from bridge_orders import write_receipt
from bridge_owner import ROOT,pin,sha,start_module
from complete_score import METRIC,METRIC_HASH
from complete_zero import FEAT,HASHES
from files29 import PEOPLE,PUBLIC_ARMS,stamp
from member_inputs import ORDERS,pinned_delivery
from public24 import C19 as C19_ROOT
from public29 import C19,C20,HARD

PAIRS=(("transport","stock"),("transport","reflection"),("transport","pooled"),
       ("transport","hard"),("transport_hard","hard"),("reflection","stock"),("pooled","stock"),("oracle","stock"))


def authenticate(order):
    receipt,bank=Path(f"public_{order}.json"),Path(f"public_{order}.npz")
    p=json.loads(receipt.read_text())
    assert p["passed"] and p["scope"]==order and not p["held_labels_opened"]
    assert p["source"]==sha(Path("source.sha256")) and p["public_source"]==sha(Path("public.sha256"))
    assert p["source_receipt"]==sha(Path("source.json")) and p["prefix"]==[31,105] and p["no_recognition"]
    assert p["stock_zero"] and p["hard_zero"] and p["stock_parent"]==list(C19[order]) and p["hard_parent"]==list(HARD[order])
    assert set(p["folds"])==set(PEOPLE) and all(sha(Path(f"prior_{k}.npz"))==v for k,v in p["folds"].items())
    assert len(p["matched_work"])==14 and not (receipt.stat().st_mode|bank.stat().st_mode)&0o222
    for a in PUBLIC_ARMS:
        assert sum(w[a]["refits"] for w in p["matched_work"].values())==126
        assert sum(w[a]["fusions"] for w in p["matched_work"].values())==1395
    pin(bank,p["bank"]["sha256"])
    parent=C19_ROOT/f"{order}.npz"
    old=C20/f"{order}.npz"
    pin(parent,C19[order][1]); pin(old,HARD[order][1])
    rows=pinned_delivery(ROOT/"runs/shuffle_perms.npz")[ORDERS.index(order)]
    with np.load(bank,allow_pickle=False) as data,np.load(parent,allow_pickle=False) as a,np.load(old,allow_pickle=False) as b:
        np.testing.assert_array_equal(data["delivery"],rows)
        np.testing.assert_array_equal(data["index"],np.arange(2655))
        y=data["scores"]
        assert y.shape==(len(PUBLIC_ARMS),2655,3) and np.isfinite(y).all()
        np.testing.assert_array_equal(y[0],a["stock|scores"]); np.testing.assert_array_equal(y[1],y[0])
        np.testing.assert_array_equal(y[5],b["hard|terminal"])
        assert p["stock_hash"]==stamp(y[0]) and p["hard_hash"]==stamp(y[5])
    return y.argmax(2),sha(receipt),sha(bank)


def evaluate(labels,y,subject,group,metric):
    pred=dict(zip(PUBLIC_ARMS,labels,strict=True))
    stock=pred["stock"]
    pred["oracle"]=np.where(pred["transport"]==y,pred["transport"],stock)
    keys=tuple(pred)
    people=sorted(set(subject)); sums=[]; counts=[]
    for h in people:
        m=subject==h
        cells=np.stack([metric.group_scores(metric.balanced_accuracy,y[m],x[m],group[m]) for x in pred.values()])
        sums.append(cells.sum(1)); counts.append(cells.shape[1])
    s,n=np.asarray(sums),np.asarray(counts)
    assert len(n)==10 and n.sum()==28
    draw=np.random.default_rng(460).integers(0,10,(4000,10))
    values=100*s.sum(0)/n.sum()
    boot=100*s[draw].sum(1)/n[draw].sum(1)[:,None]
    contrast={}
    for a,b in PAIRS:
        i,j=keys.index(a),keys.index(b)
        lo,hi=np.percentile(boot[:,i]-boot[:,j],(10,90)).tolist()
        p,q=pred[a],pred[b]
        masks={"all":np.ones_like(y,dtype=bool),**{str(c):y==c for c in (0,1,2)},**{h:subject==h for h in people}}
        moves={k:dict(rescues=int(((p==y)&(q!=y)&m).sum()),losses=int(((q==y)&(p!=y)&m).sum()),
                     changes=int(((p!=q)&m).sum())) for k,m in masks.items()}
        contrast[f"{a}_minus_{b}"]=dict(pp=float(values[i]-values[j]),paired80=[lo,hi],
                        verdict="positive" if lo>0 else "refuted" if hi<=0 else "undecided",movement=moves,
                        people_pp=dict(zip(people,(100*(s[:,i]-s[:,j])/n).tolist(),strict=True)))
    return dict(score_pct=dict(zip(keys,values.tolist(),strict=True)),contrast=contrast)


def main():
    inputs={k:authenticate(k) for k in HARD}
    lock=Path("public_before_labels.json")
    write_receipt(lock,held_labels_opened=False,
                  orders={k:dict(proof=p,bank=b,decisions=stamp(m)) for k,(m,p,b) in inputs.items()})
    pin(FEAT,HASHES[FEAT]); pin(METRIC,METRIC_HASH)
    metric=start_module("c29_public_metric",METRIC,METRIC.read_text())
    with np.load(FEAT,allow_pickle=False) as z:
        y,person,visit,iface=(z[k] for k in ("y","subject","session","interface"))
    group=np.stack((person,visit,iface),axis=1)
    results={k:evaluate(m,y,person,group,metric) for k,(m,*_) in inputs.items()}
    yes=lambda ref:all(r["contrast"][ref]["verdict"]=="positive" for r in results.values())
    out=Path("result_public.json")
    write_receipt(out,scope="FULL-SYSTEM DEV corrected held-new PROTEUS, ten reused people",before_labels=sha(lock),
                  results=results,advance=all(yes(f"transport_minus_{ref}") for ref in ("stock","reflection")),
                  above_hard=yes("transport_hard_minus_hard"))
    for order,r in results.items():
        print(order,{k:(v["pp"],v["paired80"],v["verdict"]) for k,v in r["contrast"].items()})
    print("PASS F-C29 public score",sha(out),flush=True)


if __name__=="__main__":
    main()
