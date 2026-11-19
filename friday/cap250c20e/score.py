"""Authenticate both complete native RTX arms, then score official held accuracy."""

import json
from pathlib import Path

import numpy as np

from protocol import score
from settings import ARMS, SEEDS, UPPER, schedule, sha, stamp

BASE=Path("/mnt/lustre/work/gehler/gfh527")
DATA=BASE/"neural-interfaces-2026/data/scherer2015/windows.npz"
ROWS=2130
FIELDS=("seconds","restore_s","first_s","p95_s","max_s","peak_reserved")


def anchor_path(index):
    folder="ni26_friday_frozen_rtx_20261118" if index in (0,5) else "ni26_friday_frozen_rtx_nine_20261118"
    return BASE/folder/f"rtx_cap_{index}.json"


def seal(path,data):
    with path.open("x") as f:
        json.dump(data,f,sort_keys=True,allow_nan=False)
        f.write("\n")
    path.chmod(0o444)
    return sha(path)


def verify():
    source=sha("source.sha256")
    smoke=Path("smoke.json")
    proof=json.loads(smoke.read_text())
    assert proof["passed"] and proof["source"]==source and not proof["held_labels_opened"]
    assert proof["recognized"]["stock"]["activity"]["recognized"]==10
    assert max(proof["prefix_terminal_error"],proof["cpu_pupil_error"])<=UPPER
    assert proof["protection"]["loss_rows"]>0 and proof["identity"]["adapter"]["max_move"]==0
    assert proof["stock"]["labels"]==proof["identity"]["labels"] and proof["stock"]["activity"]==proof["identity"]["activity"]
    with np.load(DATA,allow_pickle=False) as z:
        who, days, iface=[z[k] for k in ("subject","session","interface")]
    roster=sorted(set(who))
    stream=np.char.add(np.char.add(who,"|"),days)
    assert len(roster)==9 and all(len(set(iface[stream==s]))==1 for s in set(stream))
    deliveries=schedule(who,days)
    bank=np.full((2,2,ROWS),-1,dtype=np.int8)
    times={k:np.zeros((2,9,2,2)) for k in FIELDS}
    cold={}
    status, archives={},{}
    uncertainty=dict(max_upstream=0.,max_loss_weight=0.,loss_rows=0,moved_rows=0,warm_rows=0)
    for i,person in enumerate(roster):
        path=Path(f"person_{i}.json")
        assert not path.stat().st_mode & 0o222
        got=json.loads(path.read_text())
        status[path.name]=sha(path)
        anchor=anchor_path(i)
        old=json.loads(anchor.read_text())["sessions"][f"{person}|1|0"]
        archives[person]=sha(anchor)
        assert got["person"]==i and got["who"]==person and got["source"]==source and got["smoke"]==sha(smoke)
        assert got["archive"]==archives[person] and "RTX 2080 Ti" in got["gpu"] and got["torch"]=="2.14.1+cu130"
        assert not got["held_labels_opened"]
        assert got["cold"]["labels"]==old["labels"] and got["cold"]["activity"]==old["activity"]
        assert got["cold_rows"]==old["rows"]==deliveries[0][person+"|1"].tolist()
        cold[person]=dict(cold_stock={k:got["cold"][k] for k in FIELDS},
                          warm_process_prefix={a:{k:x[k] for k in FIELDS} for a,x in got["warm"].items()})
        assert len(got["sessions"])==4
        for j,seed in enumerate(SEEDS):
            for v,day in enumerate(("1","2")):
                key=f"{person}|{day}"
                rows=deliveries[seed][key]
                cell=got["sessions"][f"{key}|{seed}"]
                seq=ARMS if (i+v+j)%2==0 else ARMS[::-1]
                assert cell["order"]==list(seq) and cell["rows"]==rows.tolist()
                assert (bank[:,j,rows]==-1).all() and set(cell["runs"])==set(ARMS)
                stats=cell["protection"]
                assert 0<=max(stats["max_upstream"],stats["max_loss_weight"])<=UPPER and stats["loss_rows"]>0
                uncertainty["loss_rows"]+=stats["loss_rows"]
                for k in ("max_upstream","max_loss_weight"):
                    uncertainty[k]=max(uncertainty[k],stats[k])
                for a,arm in enumerate(ARMS):
                    rec=cell["runs"][arm]
                    act=rec["activity"]
                    assert len(rec["labels"])==len(rows) and np.isin(rec["labels"],(0,1,2)).all()
                    assert act["windows"]==len(rows) and act["recognized"]==-1 and act["errors"]==act["ft_over"]==0
                    assert act["cp_fused"]>0 and act["ft_steps"]==20*act["ft_refits"]
                    bank[a,j,rows]=rec["labels"]
                    for key2 in FIELDS:
                        times[key2][a,i,v,j]=rec[key2]
                hard=cell["runs"]["hard"]["adapter"]
                assert hard["max_move"]>.01 and hard["raw_calls"]==len(rows) and hard["warm_rows"]>0
                uncertainty["moved_rows"]+=hard["moved"]
                uncertainty["warm_rows"]+=hard["warm_rows"]
    assert (bank>=0).all() and all(np.isfinite(x).all() and (x>=0).all() for x in times.values())
    lock=seal(Path("before_labels.json"),dict(source=source,smoke=sha(smoke),roster=roster,orders=SEEDS,
              archive=archives,workers=status,stage_proofs=uncertainty,decisions=stamp(bank),held_labels_opened=False))
    return bank,times,cold,lock,roster,who,days,dict(stream=stream,interface=iface),uncertainty


def official_cells(bank,y,people,days,meta,roster):
    classes=np.zeros((2,9,2,2,3))
    for i,name in enumerate(roster):
        for v,day in enumerate(("1","2")):
            for label in (0,1,2):
                m=(people==name)&(days==day)&(y==label)
                assert m.any()
                classes[:,i,v,:,label]=100*np.mean(bank[:,:,m]==label,axis=-1)
    cells=classes.mean(-1)
    for a in range(len(ARMS)):
        for j in range(len(SEEDS)):
            official=score(y,bank[a,j],meta,np.ones(ROWS,dtype=bool))
            assert official["n_cells"]==18
            np.testing.assert_allclose(np.sort(100*np.asarray(official["cell_scores"])),
                                       np.sort(cells[a,:,:,j].ravel()),atol=1e-10,rtol=0)
    return cells,classes


def changes(bank,y,mask):
    a,b=bank[:,:,mask]
    truth=y[mask]
    hit=int(( (a!=truth)&(b==truth)).sum())
    loss=int(((a==truth)&(b!=truth)).sum())
    return dict(changed=int((a!=b).sum()),rescues=hit,regressions=loss,net=hit-loss)


def main():
    bank,times,cold,lock,roster,who,days,meta,stages=verify()
    assert sha(DATA)=="a6188799753276451235678d08e92fd79162a95ec8099dbbb85e2ed9e2701e91"
    with np.load(DATA,allow_pickle=False) as z:
        y=z["y"]
    cells,cls=official_cells(bank,y,who,days,meta,roster)
    delta=cells[1]-cells[0]
    rng=np.random.default_rng(468)
    p=rng.integers(9,size=(10000,9,1,1))
    v=rng.integers(2,size=(10000,9,2,1))
    o=rng.integers(2,size=(10000,9,2,2))
    lo,hi=np.percentile(delta[p,v,o].mean(axis=(1,2,3)),[10,90]).tolist()
    pp=dict(zip(SEEDS,delta.mean(axis=(0,1)).tolist(),strict=True))
    accuracy=lo>0 and min(pp.values())>0
    timing=times["seconds"][1]/times["seconds"][0]
    runtime=dict(median=float(np.median(timing)),worst=float(timing.max()), paired=timing.tolist(),
                 full_seconds=times["seconds"].tolist(),cold=cold,
                 warm_fields={k:x.tolist() for k,x in times.items() if k!="seconds"})
    profile=runtime["median"]<=1.05 and runtime["worst"]<=1.10
    md=dict(pp=float(delta.mean()),paired80=[lo,hi],orders_pp=pp,
            verdict="supported" if accuracy else "refuted" if hi<=0 else "undecided")
    report=dict(scope="FULL-SYSTEM EXTERNAL, RTX 2080 Ti exact Solver, reused Scherer people and fresh delivery",
                source=sha("source.sha256"),scoring=sha("scoring.sha256"),blind_lock=lock,
                absolute_pct=dict(zip(ARMS,cells.mean(axis=(1,2,3)).tolist(),strict=True)),
                contrast=md,class_pp=(cls[1]-cls[0]).mean(axis=(0,1,2)).tolist(),
                overall_changes=changes(bank,y,np.ones(ROWS,dtype=bool)),
                class_changes={str(i):changes(bank,y,y==i) for i in (0,1,2)},
                people=dict(zip(roster,delta.mean(axis=(1,2)).tolist(),strict=True)),
                people_changes={k:changes(bank,y,who==k) for k in roster},runtime=runtime,upstream=stages,
                accuracy_gate=accuracy,runtime_gate=profile,release_review=accuracy and profile)
    out=Path("result.json")
    hash_value=seal(out,report)
    print("PASS F-C20e exact paired score",hash_value,md,report["absolute_pct"],report["overall_changes"],
          runtime["median"],runtime["worst"],report["release_review"],flush=True)


if __name__=="__main__":
    main()
