"""Score complete matched native CSP branches after all predictions seal."""

import json
from pathlib import Path

import numpy as np

from readout import ARMS, stamp
from scope24 import SEEDS, schedule
from score import DATA, anchor_path, changes, official_cells, seal
from settings import sha


def open_bank():
    source=sha("source.sha256")
    smoke=json.loads(Path("smoke.json").read_text())
    assert smoke["passed"] and smoke["scope"]=="smoke" and smoke["source"]==source and not smoke["held_labels_opened"]
    assert smoke["prefix"]==[31,105] and smoke["recognized"]["prior"]["stop"]==10
    assert smoke["unrecognized"]["arms"]["temporal"]["max_q_change"]>.05
    with np.load(DATA,allow_pickle=False) as z:
        subject,visit,interface=(z[k] for k in ("subject","session","interface"))
    people=sorted(set(subject))
    assert len(people)==9
    stream=np.char.add(np.char.add(subject,"|"),visit)
    plans=schedule(subject,visit)
    bank=np.full((len(ARMS),2,len(subject)),-1,dtype=np.int8)
    signed={}
    work=dict(raw_csp_fit_rows=0,whole_features=0,csp_refits=0,csp_fusions=0,
              extra_readouts=0,upstream_seconds=0.,matched_four_csp_seconds=0.)
    for i,person in enumerate(people):
        path=Path(f"person_{i}.json")
        assert not path.stat().st_mode&0o222
        x=json.loads(path.read_text())
        parent=anchor_path(i)
        cold=json.loads(parent.read_text())["sessions"][f"{person}|1|0"]
        assert x["passed"] and x["scope"]=="produce" and not x["held_labels_opened"]
        assert x["source"]==source and x["archive"]==sha(parent) and x["person"]==person and x["index"]==i
        assert x["cold_rows"]==cold["rows"] and x["cold"]["labels"]==cold["labels"] and x["cold"]["activity"]==cold["activity"]
        signed[path.name]=sha(path)
        assert len(x["sessions"])==4
        for j,seed in enumerate(SEEDS):
            for v in (1,2):
                name=f"{person}|{v}"
                rows=plans[seed][name]
                sess=x["sessions"][f"{name}|{seed}"]
                stock=sess["arms"]["stock"]
                assert sess["rows"]==rows.tolist() and (bank[:,j,rows]==-1).all()
                assert sess["prior"]["stop"] is None and sess["prior"]["activity"]["recognized"]==-1
                assert len(sess["score_hash"])==4 and sess["score_hash"][0]==sess["score_hash"][1]
                assert stock["features"]==len(rows) and stock["fusions"]>0
                assert [r["age"] for r in stock["fitted"]]==list(range(20,len(rows),20))
                for k,arm in enumerate(ARMS):
                    item=sess["arms"][arm]
                    assert all(item[a]==stock[a] for a in ("fitted","feature_hash","refits","fusions","features","stop"))
                    if arm in ("temporal","sham"):
                        assert 0<item["extra_usage"]<=item["fusions"] and item["max_score_change"]>0.
                    preds=sess["labels"][arm]
                    assert len(preds)==len(rows) and np.isin(preds,(0,1,2)).all()
                    bank[k,j,rows]=preds
                work["raw_csp_fit_rows"]+=sum(r["n_rows"] for r in stock["fitted"])
                work["whole_features"]+=stock["features"]
                work["csp_refits"]+=stock["refits"]
                work["csp_fusions"]+=stock["fusions"]
                work["extra_readouts"]+=sess["arms"]["temporal"]["extra_usage"]
                work["upstream_seconds"]+=sess["measured_upstream_s"]
                work["matched_four_csp_seconds"]+=sess["matched_csp_s"]
    assert (bank>=0).all() and len(signed)==9
    np.testing.assert_array_equal(bank[0],bank[1])
    lock=seal(Path("before_labels.json"),dict(source=source,scoring=sha("scoring.sha256"),workers=signed,
              smoke=sha("smoke.json"),predictions=stamp(bank),orders=SEEDS,held_labels_opened=False))
    return bank,subject,visit,dict(stream=stream,interface=interface),people,work,lock


def main():
    bank,subject,visit,meta,people,work,lock=open_bank()
    with np.load(DATA,allow_pickle=False) as z:
        y=z["y"]
    stock,_,temporal,sham=bank
    oracle=np.where(temporal==y,temporal,stock)
    labels=dict(stock=stock,temporal=temporal,sham=sham,oracle=oracle)
    cells,classes={},{}
    for name in labels:
        p,c=official_cells(np.stack((stock,labels[name])),y,subject,visit,meta,people)
        cells[name],classes[name]=p[1],c[1]
    rng=np.random.default_rng(468)
    draw=(rng.integers(9,size=(10000,9,1,1)),rng.integers(2,size=(10000,9,2,1)),rng.integers(2,size=(10000,9,2,2)))
    contrasts={}
    for a,b in (("temporal","stock"),("temporal","sham"),("sham","stock"),("oracle","stock")):
        d=cells[a]-cells[b]
        limits=np.percentile(d[draw].mean(axis=(1,2,3)),(10,90)).tolist()
        orders=d.mean(axis=(0,1)).tolist()
        verdict="positive" if limits[0]>0 and min(orders)>0 else "refuted" if limits[1]<=0 else "undecided"
        contrasts[f"{a}_minus_{b}"]=dict(pp=float(d.mean()),paired80=limits,orders=orders,verdict=verdict)
    masks=dict(overall=np.ones_like(y,dtype=bool),**{person:subject==person for person in people},
               **{str(i):y==i for i in range(3)})
    decisions={k:{name:changes(np.stack((stock,p)),y,m) for name,m in masks.items()}
               for k,p in labels.items() if k!="stock"}
    result=dict(scope="FULL-SYSTEM EXTERNAL DEV matched live cap250 upstream, nine reused Scherer people",
                before_labels=lock,orders=SEEDS,work=work,contrast=contrasts,decision_changes=decisions,
                score_pct={k:float(x.mean()) for k,x in cells.items()},
                people_pp={k:dict(zip(people,(x-cells["stock"]).mean(axis=(1,2)).tolist(),strict=True)) for k,x in cells.items()},
                class_pp={k:(x-classes["stock"]).mean(axis=(0,1,2)).tolist() for k,x in classes.items()},
                advances=all(contrasts[f"temporal_minus_{b}"]["verdict"]=="positive" for b in ("stock","sham")))
    digest=seal(Path("result_external.json"),result)
    print("PASS F-C24 external",digest,contrasts,result["advances"],flush=True)


if __name__=="__main__":
    main()
