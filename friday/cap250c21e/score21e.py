"""Score the independently qualified exact RTX worker pairs."""

import json
from pathlib import Path

import numpy as np

from scope import ARMS, SEEDS, schedule
from score import DATA, FIELDS, anchor_path, changes, official_cells, seal
from settings import UPPER, sha, stamp


def authenticate():
    root=sha("source.sha256")
    smoke=json.loads(Path("smoke.json").read_text())
    assert smoke["source"]==root and smoke["passed"] and not smoke["held_labels_opened"]
    assert smoke["identity_proof"]["exact_gate"] and smoke["identity_proof"]["stage_error"]<=UPPER
    assert smoke["fbl2"]["hook"]["warm_rows"]>0 and smoke["recognized"]["fbl2"]["activity"]["recognized"]==10
    assert smoke["recognized"]["fbl2"]["hook"]["route_share"]==1/17
    with np.load(DATA,allow_pickle=False) as z:
        subject,visit,iface=z["subject"],z["session"],z["interface"]
    people=sorted(set(subject))
    group=np.char.add(np.char.add(subject,"|"),visit)
    orders=schedule(subject,visit)
    matrix=np.full((2,2,len(subject)),-1,dtype=np.int8)
    receipts={}
    active, count_rows, max_drift=0,0,0.
    for i,p in enumerate(people):
        path=Path(f"person_{i}.json")
        x=json.loads(path.read_text())
        anchor=anchor_path(i)
        old=json.loads(anchor.read_text())["sessions"][f"{p}|1|0"]
        assert x["archive"]==sha(anchor) and x["source"]==root and x["person"]==p and x["index"]==i
        assert x["passed"] and not x["held_labels_opened"] and x["scope"]=="produce"
        assert not path.stat().st_mode&0o222
        assert x["cold_rows"]==old["rows"] and x["cold"]["labels"]==old["labels"]
        assert x["cold"]["activity"]==old["activity"]
        receipts[path.name]=sha(path)
        assert len(x["results"])==4
        for j,seed in enumerate(SEEDS):
            for v,d in enumerate((1,2)):
                name=f"{p}|{d}"
                rows=orders[seed][name]
                item=x["results"][f"{name}|{seed}"]
                assert item["rows"]==rows.tolist() and (matrix[:,j,rows]==-1).all()
                seq=ARMS if (i+j+v)%2==0 else ARMS[::-1]
                assert item["order"]==list(seq)
                pr=item["proof"]
                assert pr["exact_gate"] and 0<=max(pr["max_upstream"],pr["max_loss_weight"])<=UPPER
                assert pr["loss_rows"]>0
                max_drift=max(pr["max_upstream"],max_drift)
                count_rows+=pr["loss_rows"]
                assert not item["runs"]["fbl2"]["route"]["routed"]
                for k,arm in enumerate(ARMS):
                    r=item["runs"][arm]
                    a=r["activity"]
                    assert r["hook"]["calls"]==a["windows"]==len(rows) and r["hook"]["warm_rows"]>0
                    assert a["errors"]==a["ft_over"]==0 and a["ft_refits"]*20==a["ft_steps"]
                    assert len(r["labels"])==len(rows) and np.isin(r["labels"],(0,1,2)).all()
                    matrix[k,j,rows]=r["labels"]
                active+=item["runs"]["fbl2"]["hook"]["warm_rows"]
    assert (matrix>=0).all() and len(receipts)==9
    lock=seal(Path("before_labels.json"),dict(source=root,scoring=sha("scoring.sha256"),stock=receipts,
              smoke=sha("smoke.json"),decisions=stamp(matrix),orders=SEEDS,held_labels_opened=False))
    return matrix,subject,visit,dict(stream=group,interface=iface),people,lock,dict(
        warm_rows=active,loss_rows=count_rows,max_upstream=max_drift)


def main():
    matrix,subject,visit,meta,people,lock,proof=authenticate()
    with np.load(DATA,allow_pickle=False) as z:
        labels=z["y"]
    cells,per_class=official_cells(matrix,labels,subject,visit,meta,people)
    d=cells[1]-cells[0]
    rng=np.random.default_rng(468)
    p=rng.integers(9,size=(10000,9,1,1))
    v=rng.integers(2,size=(10000,9,2,1))
    o=rng.integers(2,size=(10000,9,2,2))
    bound=np.percentile(d[p,v,o].mean(axis=(1,2,3)),[10,90]).tolist()
    by_order=dict(zip(SEEDS,d.mean(axis=(0,1)).tolist(),strict=True))
    gain=bound[0]>0 and min(by_order.values())>0
    verdict="supported" if gain else "refuted" if bound[1]<=0 else "undecided"
    result=dict(scope="FULL-SYSTEM EXTERNAL independent exact RTX 2080Ti, reused Scherer people",
                score_pct=dict(zip(ARMS,cells.mean(axis=(1,2,3)).tolist(),strict=True)),before_labels=lock,
                contrast=dict(pp=float(d.mean()),paired80=bound,orders=by_order,verdict=verdict),
                overall=changes(matrix,labels,np.ones(len(labels),dtype=bool)),upstream=proof,
                classes={str(k):changes(matrix,labels,labels==k) for k in (0,1,2)},
                class_pp=(per_class[1]-per_class[0]).mean(axis=(0,1,2)).tolist(),
                people_pp=dict(zip(people,d.mean(axis=(1,2)).tolist(),strict=True)),
                per_person={p:changes(matrix,labels,subject==p) for p in people})
    digest=seal(Path("result.json"),result)
    print("PASS F-C21e independently paired native score",digest,result["score_pct"],
          result["contrast"],result["overall"],flush=True)


if __name__=="__main__":
    main()
