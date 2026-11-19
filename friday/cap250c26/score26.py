"""Authenticate three native optimizer arms before opening Scherer gold."""

import json
from pathlib import Path

import numpy as np

from scope26 import ARMS,SEEDS,schedule
from score import DATA,anchor_path,changes,official_cells,seal
from settings import sha,stamp

HISTORY=("age","labels","confidence","gen_before","gen_after","frozen","raw")
PROTECTED=("windows","recognized","ft_refits","ft_relabels","ft_steps","ft_capped","ft_over","errors")


def verify():
    source=sha("source.sha256")
    witness=json.loads(Path("smoke.json").read_text())
    assert witness["passed"] and witness["source"]==source and witness["scope"]=="smoke"
    assert not witness["held_labels_opened"] and max(witness["zero"].values())<=5e-5 and witness["prefix"]<=5e-5
    assert witness["recognized"]["stock"]["activity"]["recognized"]==10
    assert not witness["recognized"]["peer"]["events"] and witness["identity"]["fit"]["consumes"]==40
    with np.load(DATA,allow_pickle=False) as data:
        person,visit,i=(data[k] for k in ("subject","session","interface"))
    people=sorted(set(person))
    plans=schedule(person,visit)
    bank=np.full((3,2,len(person)),-1,dtype=np.int8)
    receipts={}
    work=dict(windows=0,ft_refits=0,ft_steps=0,fit_rows=0,
              arms={k:dict(cp_refits=0,cp_fused=0,touched=0,mass_error=0.,max_weight_move=0.,
                           family_difference=0.,minimum_factor=1.,maximum_factor=1.,seconds=0.) for k in ARMS})
    for ident,p in enumerate(people):
        path=Path(f"person_{ident}.json")
        assert not path.stat().st_mode&0o222
        x=json.loads(path.read_text())
        anchor=anchor_path(ident)
        archive=json.loads(anchor.read_text())["sessions"][f"{p}|1|0"]
        assert x["passed"] and x["source"]==source and x["scope"]=="produce" and not x["held_labels_opened"]
        assert x["index"]==ident and x["person"]==p and x["archive"]==sha(anchor) and "RTX 2080 Ti" in x["device"]
        assert x["cold_rows"]==archive["rows"] and x["cold"]["labels"]==archive["labels"]
        assert x["cold"]["activity"]==archive["activity"] and len(x["sessions"])==4
        assert max(x["zero"].values())<=5e-5 and x["zero_scope"]["consumes"]==8 and x["zero_scope"]["touched"]==0
        receipts[path.name]=sha(path)
        for j,seed in enumerate(SEEDS):
            for d in (1,2):
                name=f"{p}|{d}"
                rows=plans[seed][name]
                sess=x["sessions"][f"{name}|{seed}"]
                shift=(ident+j+d)%3
                assert sess["rows"]==rows.tolist() and sess["order"]==list(ARMS[shift:]+ARMS[:shift])
                assert (bank[:,j,rows]==-1).all()
                baseline=sess["runs"]["stock"]
                for k,arm in enumerate(ARMS):
                    a=sess["runs"][arm]
                    f=a["fit"]
                    c=a["activity"]
                    assert all(c[q]==baseline["activity"][q] for q in PROTECTED)
                    assert c["windows"]==len(rows) and c["recognized"]==-1 and c["errors"]==c["ft_over"]==0
                    assert c["ft_steps"]==20*c["ft_refits"]==20*f["refits"]
                    assert [e["age"] for e in a["events"]]==list(range(20,len(rows),10))
                    assert a["protected"]==baseline["protected"] and a["gate"]["gates"]==baseline["gate"]["gates"]
                    assert a["labels"][:20]==baseline["labels"][:20]
                    for left,right in zip(a["events"],baseline["events"],strict=True):
                        assert all(left[q]==right[q] for q in HISTORY)
                    if arm!="stock":
                        assert f["consumes"]==4*f["refits"] and f["active"]>=5 and f["mass_error"]<=2e-7
                        assert sess["movement"][arm]["warm"]>1e-5 and sess["movement"][arm]["stages"]>1e-4
                        assert f["family_difference"]>0 and f["touched"]>0
                    pred=a["labels"]
                    assert len(pred)==len(rows) and np.isin(pred,(0,1,2)).all()
                    bank[k,j,rows]=pred
                    w=work["arms"][arm]
                    for q in ("cp_refits","cp_fused"):
                        w[q]+=c[q]
                    w["seconds"]+=a["seconds"]
                    w["touched"]+=f["touched"]
                    for q in ("mass_error","max_weight_move","family_difference"):
                        w[q]=max(w[q],f[q])
                    w["minimum_factor"]=min(w["minimum_factor"],f["factors"][0]); w["maximum_factor"]=max(w["maximum_factor"],f["factors"][1])
                work["windows"]+=len(rows)
                work["ft_refits"]+=baseline["activity"]["ft_refits"]
                work["ft_steps"]+=baseline["activity"]["ft_steps"]
                work["fit_rows"]+=baseline["fit"]["rows"]
    assert (bank>=0).all() and work["windows"]==4260
    lock=seal(Path("before_labels.json"),dict(source=source,scoring=sha("scoring.sha256"),workers=receipts,
              smoke=sha("smoke.json"),orders=SEEDS,decisions=stamp(bank),held_labels_opened=False))
    return bank,person,visit,dict(stream=np.char.add(np.char.add(person,"|"),visit),interface=i),people,work,lock


def main():
    bank,person,visit,meta,people,work,lock=verify()
    with np.load(DATA,allow_pickle=False) as z:
        truth=z["y"]
    stock,peer,sham=bank
    predictions=dict(zip(ARMS,bank,strict=True),oracle=np.where(peer==truth,peer,stock))
    means,classes={},{}
    for arm,p in predictions.items():
        m,c=official_cells(np.stack((stock,p)),truth,person,visit,meta,people)
        means[arm],classes[arm]=m[1],c[1]
    rng=np.random.default_rng(468)
    draws=(rng.integers(9,size=(10000,9,1,1)),rng.integers(2,size=(10000,9,2,1)),rng.integers(2,size=(10000,9,2,2)))
    pairs=(("peer","stock"),("peer","sham"),("sham","stock"),("oracle","stock"))
    contrasts={}
    movements={}
    masks=dict(overall=np.ones_like(truth,dtype=bool),**{p:person==p for p in people},
               **{str(k):truth==k for k in (0,1,2)})
    for a,b in pairs:
        diff=means[a]-means[b]
        limits=np.percentile(diff[draws].mean(axis=(1,2,3)),(10,90)).tolist()
        orders=diff.mean(axis=(0,1)).tolist()
        name=f"{a}_minus_{b}"
        contrasts[name]=dict(pp=float(diff.mean()),paired80=limits,orders=orders,
                   verdict="positive" if limits[0]>0 and min(orders)>0 else "refuted" if limits[1]<=0 else "undecided",
                   people=dict(zip(people,diff.mean(axis=(1,2)).tolist(),strict=True)),
                   class_pp=(classes[a]-classes[b]).mean(axis=(0,1,2)).tolist())
        movements[name]={key:changes(np.stack((predictions[b],predictions[a])),truth,mask) for key,mask in masks.items()}
    score=dict(scope="FULL-SYSTEM EXTERNAL native independent optimizer branches, nine reused Scherer people",
               orders=SEEDS,before_labels=lock,work=work,contrast=contrasts,movement=movements,
               absolute_pct={k:float(v.mean()) for k,v in means.items()},
               advances=all(contrasts[f"peer_minus_{k}"]["verdict"]=="positive" for k in ("stock","sham")))
    digest=seal(Path("result.json"),score)
    print("PASS F-C26",digest,score["absolute_pct"],contrasts,score["advances"],flush=True)


if __name__=="__main__":
    main()
