"""Authenticate and score both corrected public instruments after sealing decisions."""

import json
from pathlib import Path

import numpy as np

from bridge_orders import write_receipt
from bridge_owner import ROOT, pin, sha, start_module
from complete_score import METRIC, METRIC_HASH
from complete_zero import FEAT, HASHES
from member_inputs import ORDERS, pinned_delivery
from public24 import C19, PINS
from readout import ARMS, stamp


def bank(order):
    path=Path(f"public_{order}.json")
    receipt=json.loads(path.read_text())
    assert receipt["passed"] and not receipt["held_labels_opened"] and receipt["scope"]==order
    assert receipt["source"]==sha("source.sha256") and receipt["public_source"]==sha("public.sha256")
    assert receipt["stock_bitwise"] and receipt["identity_bitwise"] and receipt["no_recognition"]
    assert receipt["prefix"]==[31,105] and receipt["warm_steps"]==9600 and len(receipt["match_proofs"])==14
    assert receipt["archived"]==list(PINS[order])
    parent=C19/f"{order}.npz"
    pin(parent,PINS[order][1])
    blob=Path(f"public_{order}.npz")
    pin(blob,receipt["bank"]["sha256"])
    assert not path.stat().st_mode&0o222 and not blob.stat().st_mode&0o222
    with np.load(blob,allow_pickle=False) as z, np.load(parent,allow_pickle=False) as original:
        np.testing.assert_array_equal(z["delivery"],pinned_delivery(ROOT/"runs/shuffle_perms.npz")[ORDERS.index(order)])
        np.testing.assert_array_equal(z["index"],np.arange(2655))
        for key,old in (("raw","raw"),("count","filtered")):
            np.testing.assert_array_equal(z[key],original[f"stock|{old}"])
            assert stamp(z[key])==receipt[key]
        scores=z["scores"]
        assert scores.shape==(4,2655,3) and np.isfinite(scores).all()
        for a in range(2):
            np.testing.assert_array_equal(scores[a],original["stock|scores"])
    return scores.argmax(2),sha(path),sha(blob)


def analyze(pred,y,groups,people,metric):
    real=dict(zip(ARMS,pred,strict=True))
    base=real["stock"]
    real["oracle"]=np.where(real["temporal"]==y,real["temporal"],base)
    names=sorted(set(people))
    sums,counts=[],[]
    for name in names:
        use=people==name
        scores=np.stack([metric.group_scores(metric.balanced_accuracy,y[use],p[use],groups[use]) for p in real.values()])
        counts.append(scores.shape[1]); sums.append(scores.sum(1))
    n,s=np.asarray(counts),np.asarray(sums)
    assert len(n)==10 and n.sum()==28
    draws=np.random.default_rng(460).integers(0,10,(4000,10))
    u=100*s.sum(0)/n.sum()
    b=100*s[draws].sum(1)/n[draws].sum(1)[:,None]
    plus={}
    for a,ref in (("temporal","stock"),("temporal","sham"),("sham","stock"),("oracle","stock")):
        i,j=(list(real).index(k) for k in (a,ref))
        lo,hi=np.percentile(b[:,i]-b[:,j],(10,90)).tolist()
        plus[f"{a}_minus_{ref}"]=dict(pp=float(u[i]-u[j]),paired80=[lo,hi],
                    verdict="positive" if lo>0 else "refuted" if hi<=0 else "undecided")
    def change(p,rows):
        return dict(rescues=int(((base!=y)&(p==y)&rows).sum()),losses=int(((base==y)&(p!=y)&rows).sum()),
                    changed=int(((base!=p)&rows).sum()))
    return dict(score_pct=dict(zip(real,u.tolist(),strict=True)),contrasts=plus,
                movement={k:change(p,np.ones(len(y),dtype=bool)) for k,p in real.items()},
                class_changes={k:{c:change(p,y==c) for c in (0,1,2)} for k,p in real.items()},
                people_pp={k:dict(zip(names,(100*(s[:,i]-s[:,0])/n).tolist(),strict=True)) for i,k in enumerate(real)})


def main():
    pins={order:bank(order) for order in PINS}
    locked=Path("public_before_labels.json").resolve()
    write_receipt(locked,source=sha("public.sha256"),held_labels_opened=False,
                  orders={order:dict(worker=worker,bank=blob,labels=stamp(pred)) for order,(pred,worker,blob) in pins.items()})
    pin(FEAT,HASHES[FEAT]); pin(METRIC,METRIC_HASH)
    metric=start_module("f24_metric",METRIC,METRIC.read_text())
    with np.load(FEAT,allow_pickle=False) as z:
        y,person,visit,interface=(z[k] for k in ("y","subject","session","interface"))
    group=np.stack((person,visit,interface),axis=1)
    scored={order:analyze(pred,y,group,person,metric) for order,(pred,*_) in pins.items()}
    out=Path("result_public.json").resolve()
    write_receipt(out,scope="FULL-SYSTEM DEV corrected held-new PROTEUS, reused people and two instruments",
                  before_labels=sha(locked),results=scored,
                  advances=all(r["contrasts"][f"temporal_minus_{c}"]["verdict"]=="positive"
                               for r in scored.values() for c in ("stock","sham")))
    print("PASS F-C24 public score",sha(out),{k:r["contrasts"] for k,r in scored.items()},flush=True)


if __name__=="__main__":
    main()
