"""Score both conditional native outer histories after the nine banks seal."""

import json
from pathlib import Path

import numpy as np

from score import anchor_path, changes, official_cells, seal
from settings import ARMS, SEEDS, schedule, sha, stamp

DATA=Path("/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/data/scherer2015/windows.npz")


def full_bank():
    source=sha("source.sha256")
    smoke=json.loads(Path("smoke.json").read_text())
    assert smoke["passed"] and smoke["mode"]=="smoke" and not smoke["held_labels_opened"]
    assert smoke["source"]==source and smoke["shared_protected"] and smoke["distinct_outer"]
    assert smoke["sessions"]["P10-3"]["stop"]==10
    assert all(s["zero_exact"] for s in smoke["sessions"].values())
    with np.load(DATA,allow_pickle=False) as z:
        subject,day,iface=z["subject"],z["session"],z["interface"]
    people=sorted(set(subject))
    names=np.char.add(np.char.add(subject,"|"),day)
    plans=schedule(subject,day)
    assert len(people)==9 and all(len(set(iface[names==s]))==1 for s in set(names))
    matrix=np.full((2,2,2130),-1,dtype=np.int8)
    signed={}
    scope=dict(max_mix=0.,warm=0,refits=0,stock_fuses=0,candidate_fuses=0,owner_seconds=0.,mirror_seconds=0.)
    for i,person in enumerate(people):
        path=Path(f"person_{i}.json")
        pupil=Path(f"pupils_produce_{i}.npz")
        assert not (path.stat().st_mode|pupil.stat().st_mode)&0o222
        got=json.loads(path.read_text())
        parent=anchor_path(i)
        archived=json.loads(parent.read_text())["sessions"][f"{person}|1|0"]
        assert got["passed"] and got["mode"]=="produce" and not got["held_labels_opened"]
        assert got["source"]==source and got["pupil"]==sha(pupil) and got["archive"]==sha(parent)
        assert got["who"]==person and got["person"]==i and "RTX 2080 Ti" in got["device"]
        assert got["shared_protected"] and got["distinct_outer"] and got["original_labels"]==archived["labels"]
        signed[path.name]=sha(path)
        for j,seed in enumerate(SEEDS):
            for v in (1,2):
                stream=f"{person}|{v}"
                p=plans[seed][stream]
                r=got["sessions"][f"{stream}|{seed}"]
                assert p.tolist()==r["rows"] and r["stop"] is None and (matrix[:,j,p]<0).all()
                act=r["activity"]
                assert act["windows"]==len(p) and act["recognized"]==-1 and act["errors"]==act["ft_over"]==0
                assert act["ft_steps"]==20*act["ft_refits"] and r["max_mix"]>1e-6 and r["warm_mix"]>0
                assert r["candidate_fit_ages"]==list(range(20,len(p),20))
                assert r["candidate_refits"]==len(r["candidate_fit_ages"])
                for k,arm in enumerate(ARMS):
                    assert len(r[arm])==len(p) and np.isin(r[arm],(0,1,2)).all()
                    matrix[k,j,p]=r[arm]
                scope["max_mix"]=max(scope["max_mix"],r["max_mix"])
                scope["warm"]+=r["warm_mix"]
                scope["refits"]+=act["ft_refits"]
                scope["stock_fuses"]+=act["cp_fused"]
                scope["candidate_fuses"]+=r["candidate_fuses"]
                scope["owner_seconds"]+=r["owner_s"]
                scope["mirror_seconds"]+=r["mirror_s"]
    assert (matrix>=0).all()
    lock=seal(Path("before_labels.json"),dict(source=source,scoring=sha("scoring.sha256"),workers=signed,
             prediction_sha=stamp(matrix),smoke=sha("smoke.json"),people=people,orders=SEEDS,held_labels_opened=False))
    return matrix,subject,day,dict(stream=names,interface=iface),people,scope,lock


def main():
    bank,subjects,days,meta,people,scope,lock=full_bank()
    with np.load(DATA,allow_pickle=False) as z:
        truth=z["y"]
    cells,classes=official_cells(bank,truth,subjects,days,meta,people)
    diff=cells[1]-cells[0]
    rng=np.random.default_rng(468)
    p=rng.integers(9,size=(10000,9,1,1))
    v=rng.integers(2,size=(10000,9,2,1))
    o=rng.integers(2,size=(10000,9,2,2))
    bounds=np.percentile(diff[p,v,o].mean(axis=(1,2,3)),[10,90]).tolist()
    orders=dict(zip(SEEDS,diff.mean(axis=(0,1)).tolist(),strict=True))
    verdict=("supported" if bounds[0]>0 and min(orders.values())>0 else "refuted" if bounds[1]<=0 else "undecided")
    report=dict(scope="FULL-SYSTEM EXTERNAL conditional on the exact same stock upstream, RTX proxy",
                before_labels=lock,absolute_pct=dict(zip(ARMS,cells.mean(axis=(1,2,3)).tolist(),strict=True)),
                contrast=dict(pp=float(diff.mean()),paired80=bounds,orders_pp=orders,verdict=verdict),
                class_pp=(classes[1]-classes[0]).mean(axis=(0,1,2)).tolist(),
                decisions=changes(bank,truth,np.ones(len(truth),dtype=bool)),
                per_class={str(i):changes(bank,truth,truth==i) for i in (0,1,2)},
                people_pp=dict(zip(people,diff.mean(axis=(1,2)).tolist(),strict=True)),
                per_person={a:changes(bank,truth,subjects==a) for a in people},native_work=scope)
    result=seal(Path("result.json"),report)
    print("PASS F-C20s score",result,report["contrast"],report["absolute_pct"],report["decisions"],flush=True)


if __name__=="__main__":
    main()
