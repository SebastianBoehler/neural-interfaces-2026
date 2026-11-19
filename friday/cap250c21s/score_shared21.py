"""Score complete fixed-native-input FBLight pairs after sealing all nine people."""

import json
from pathlib import Path

import numpy as np

from scope import ARMS, SEEDS, schedule
from score import DATA, anchor_path, changes, official_cells, seal
from settings import sha, stamp


def full_bank():
    source=sha("source.sha256")
    smoke=json.loads(Path("smoke.json").read_text())
    assert smoke["passed"] and smoke["scope"]=="smoke" and not smoke["held_labels_opened"]
    assert smoke["source"]==source and "RTX 2080 Ti" in smoke["device"]
    recognized=smoke["recognized"]
    assert recognized["protected"]["stop"]==10 and recognized["route"]["routed"]==21
    assert smoke["unrecognized"]["hook"]["warm_rows"]==100
    assert all(smoke[k]["identity_exact"] and smoke[k]["prefix_exact"] for k in ("recognized","unrecognized"))
    with np.load(DATA,allow_pickle=False) as z:
        subject,day,iface=z["subject"],z["session"],z["interface"]
    people=sorted(set(subject))
    group=np.char.add(np.char.add(subject,"|"),day)
    assert len(people)==9 and len(set(group))==18 and all(len(set(iface[group==g]))==1 for g in set(group))
    orders=schedule(subject,day)
    matrix=np.full((2,2,len(subject)),-1,dtype=np.int8)
    signed={}
    summary=dict(warm_rows=0,prior_fit_rows=0,stock_csp_fusions=0,fbl2_csp_fusions=0,identity_rows=0,
                 max_mix=0.,max_terminal=0.)
    for i,person in enumerate(people):
        file=Path(f"person_{i}.json")
        assert not file.stat().st_mode&0o222
        x=json.loads(file.read_text())
        path=anchor_path(i)
        archive=json.loads(path.read_text())["sessions"][f"{person}|1|0"]
        assert x["passed"] and x["scope"]=="produce" and not x["held_labels_opened"]
        assert x["source"]==source and x["archive"]==sha(path) and "RTX 2080 Ti" in x["device"]
        assert x["person"]==person and x["index"]==i and len(x["results"])==4
        assert x["cold_rows"]==archive["rows"] and x["cold"]["labels"]==archive["labels"]
        assert x["cold"]["activity"]==archive["activity"]
        signed[file.name]=sha(file)
        for j,seed in enumerate(SEEDS):
            for v in (1,2):
                key=f"{person}|{v}"
                pos=orders[seed][key]
                r=x["results"][f"{key}|{seed}"]
                p,h=r["protected"],r["hook"]
                a=p["activity"]
                n=len(pos)
                assert r["identity_exact"] and r["rows"]==pos.tolist() and (matrix[:,j,pos]==-1).all()
                assert a["windows"]==h["calls"]==h["moved"]==n and h["warm_rows"]>=n-20
                assert p["stop"] is None and a["recognized"]==-1 and r["route"]["routed"]==0
                assert p["fit_ages"]==list(range(20,n,10)) and p["prior_rows"]==sum(p["fit_ages"])
                assert a["errors"]==a["ft_over"]==0 and a["ft_steps"]==20*a["ft_refits"]
                assert h["route_share"]==1/17 and h["max_mix_delta"]>.001 and r["terminal_change"]>1e-4
                assert r["trial_fits"]==list(range(20,n,20)) and r["trial_fuses"]>0
                for k,arm in enumerate(ARMS):
                    vec=r[arm]
                    assert len(vec)==n and np.isin(vec,(0,1,2)).all()
                    matrix[k,j,pos]=vec
                summary["warm_rows"]+=h["warm_rows"]
                summary["identity_rows"]+=n
                summary["prior_fit_rows"]+=p["prior_rows"]
                summary["stock_csp_fusions"]+=p["stock_fuses"]
                summary["fbl2_csp_fusions"]+=r["trial_fuses"]
                summary["max_mix"]=max(summary["max_mix"],h["max_mix_delta"])
                summary["max_terminal"]=max(summary["max_terminal"],r["terminal_change"])
    assert (matrix>=0).all() and summary["identity_rows"]==4260
    lock=seal(Path("before_labels.json"),dict(source=source,scoring=sha("scoring.sha256"),workers=signed,
              prediction_sha=stamp(matrix),smoke=sha("smoke.json"),people=people,orders=SEEDS,held_labels_opened=False))
    return matrix,subject,day,dict(stream=group,interface=iface),people,summary,lock


def main():
    matrix,subject,day,meta,people,scope,lock=full_bank()
    with np.load(DATA,allow_pickle=False) as z:
        y=z["y"]
    cells,classes=official_cells(matrix,y,subject,day,meta,people)
    diff=cells[1]-cells[0]
    rng=np.random.default_rng(468)
    p=rng.integers(9,size=(10000,9,1,1))
    v=rng.integers(2,size=(10000,9,2,1))
    o=rng.integers(2,size=(10000,9,2,2))
    bounds=np.percentile(diff[p,v,o].mean(axis=(1,2,3)),[10,90]).tolist()
    orders=dict(zip(SEEDS,diff.mean(axis=(0,1)).tolist(),strict=True))
    verdict="supported" if bounds[0]>0 and min(orders.values())>0 else "refuted" if bounds[1]<=0 else "undecided"
    report=dict(scope="FULL-SYSTEM EXTERNAL conditional on the same native RTX upstream",
                before_labels=lock,score_pct=dict(zip(ARMS,cells.mean(axis=(1,2,3)).tolist(),strict=True)),
                contrast=dict(pp=float(diff.mean()), paired80=bounds, orders_pp=orders, verdict=verdict),
                decisions=changes(matrix,y,np.ones(len(y),dtype=bool)),
                by_order={seed:dict(zip(ARMS,cells[:,:,:,j].mean(axis=(1,2)).tolist(),strict=True))
                          for j,seed in enumerate(SEEDS)},
                people_pp=dict(zip(people,diff.mean(axis=(1,2)).tolist(),strict=True)),
                per_person={person:changes(matrix,y,subject==person) for person in people},
                class_pp=(classes[1]-classes[0]).mean(axis=(0,1,2)).tolist(),
                per_class={str(i):changes(matrix,y,y==i) for i in (0,1,2)},native_work=scope)
    digest=seal(Path("result.json"),report)
    print("PASS F-C21s score",digest,report["score_pct"],report["contrast"],report["decisions"],flush=True)


if __name__=="__main__":
    main()
