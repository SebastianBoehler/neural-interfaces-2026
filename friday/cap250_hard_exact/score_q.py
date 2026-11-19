"""Score independent complete Solver archive decisions after authenticating the blind bank."""
import json
from pathlib import Path

import numpy as np

from fixture import DATA_SHA, SEEDS, schedule
from protocol import score
from score import DATA, anchor_path, seal
from settings import sha, stamp

TRIAL = "6a569180d343c038558fe7eccb7903a36e2bdf694af0df3687bff3b17049fbcc"
STOCK = "9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed"
ACTORS = "0292022abd39067cefebd0e3a0bd8f1834a04bfc67d7d41081cfe6abaae8ebd2"
ARMS = ("stock_A", "stock_B", "hard")
ROWS = 2130


def compare(cells, classes):
    rng = np.random.default_rng(625)
    i = rng.integers(9,size=(14000,9,1,1))
    v = rng.integers(2,size=(14000,9,2,1))
    o = rng.integers(2,size=(14000,9,2,2))

    def summary(delta):
        return dict(pp=float(delta.mean()), paired80=np.percentile(delta[i,v,o].mean((1,2,3)),[10,90]).tolist(),
                    order_pp=list(map(float,delta.mean((0,1)))))

    a,b,h = cells
    delta = h-(a+b)/2
    primary=summary(delta)
    aa=summary(b-a)
    other=dict(vs_stock_A=summary(h-a),vs_stock_B=summary(h-b))
    success=(primary["paired80"][0]>0 and min(primary["order_pp"])>0
             and all(min(x["order_pp"])>0 for x in other.values()))
    ac,bc,hc=classes
    class_pp=[summary((hc-(ac+bc)/2)[...,c]) for c in range(3)]
    return dict(hard_vs_mean_stock=primary,**other,stock_B_vs_stock_A=aa,
                class_effects=class_pp,advanced=bool(success),person_pp=delta.mean((1,2)).tolist())


def qualified():
    assert sha(DATA)==DATA_SHA
    source=sha("source.sha256")
    smoke=json.loads(Path("smoke.json").read_text())
    assert smoke["passed"] and smoke["mode"]=="smoke" and smoke["source"]==source
    assert smoke["archive"]==TRIAL and smoke["stock"]==STOCK and smoke["actors"]==ACTORS
    assert not smoke["held_labels_opened"] and set(smoke["results"])=={"Scherer2015/G|1|0","P10-3"}
    assert all(r["reset"]["reused"]["exact_cpu_actor_reset"] for r in smoke["results"].values())
    with np.load(DATA,allow_pickle=False) as z:
        who,visit,interface=[z[k] for k in ("subject","session","interface")]
    stream=np.char.add(np.char.add(who,"|"),visit)
    people=sorted(set(who))
    plans=schedule(who,visit)
    bank=np.full((3,2,ROWS),-1,np.int8)
    pins={}
    variation={}
    for index,p in enumerate(people):
        path=Path(f"person_{index}.json")
        assert not path.stat().st_mode&0o222
        receipt=json.loads(path.read_text())
        assert receipt["passed"] and receipt["mode"]=="produce" and receipt["index"]==index and receipt["who"]==p
        assert receipt["source"]==source and not receipt["held_labels_opened"]
        assert receipt["archive"]==TRIAL and receipt["stock"]==STOCK and receipt["actors"]==ACTORS
        assert "RTX 2080 Ti" in receipt["worker"]
        archive=anchor_path(index)
        canonical=json.loads(archive.read_text())["sessions"][f"{p}|1|0"]
        assert receipt["cold_archive"]==sha(archive) and receipt["cold_rows"]==canonical["rows"]
        assert receipt["cold_labels"]==canonical["labels"] and receipt["cold_work"]==canonical["activity"]
        pins[path.name]=sha(path)
        assert len(receipt["results"])==4
        local={}
        for i,seed in enumerate(SEEDS):
            for day in ("1","2"):
                key=f"{p}|{day}"
                rows=plans[seed][key]
                entry=receipt["results"][f"{key}|{seed}"]
                assert entry["reversed"]==((i+int(day)-1)%2==1)
                assert entry["rows"]==rows.tolist() and (bank[:,i,rows]<0).all()
                proof=entry["proof"]
                assert len(entry["labels"])==3 and all(len(y)==len(rows) for y in entry["labels"])
                assert proof["calls"]==len(rows) and proof["sha_pupils_exact"] and proof["warm_rows"]>0
                assert proof["exact_private_tail"] and proof["exact_same_input_prefix"]
                assert proof["exact_stock_and_candidate_native_inputs"]
                assert all(x["windows"]==len(rows) and x["recognized"]==-1 and x["errors"]==x["ft_over"]==0
                           for x in entry["work"])
                bank[:,i,rows]=entry["labels"]
                local[f"{day}|{seed}"]=dict(A_A=entry["stock_AA"]["terminal_max"],
                                            A_tr=entry["max_protected_trial"][0],
                                            B_tr=entry["max_protected_trial"][1])
        variation[p]=local
    assert np.isin(bank,(0,1,2)).all()
    proof=seal(Path("before_labels.json"),dict(held_labels_opened=False, source=source, scoring=sha("scoring.sha256"),
               archive=TRIAL, stock=STOCK, actors=ACTORS, smoke=sha("smoke.json"), pins=pins, predictions=stamp(bank)))
    return bank,who,visit,interface,stream,people,variation,proof


def run():
    bank,who,visit,interface,stream,people,variation,signed=qualified()
    with np.load(DATA,allow_pickle=False) as z:
        gold=z["y"]
    c=np.full((3,9,2,2,3), np.nan)
    for i,p in enumerate(people):
        for j,day in enumerate(("1","2")):
            for label in range(3):
                take=(who==p)&(visit==day)&(gold==label)
                assert take.any()
                c[:,i,j,:,label]=100*np.mean(bank[:,:,take]==label,axis=-1)
    cells=c.mean(-1)
    for k in range(3):
        for j in range(2):
            ref=score(gold,bank[k,j],dict(stream=stream,interface=interface),np.ones(ROWS,bool))
            assert ref["n_cells"]==18
            np.testing.assert_allclose(np.sort(100*np.asarray(ref["cell_scores"])), np.sort(cells[k,:,:,j].ravel()),
                                       rtol=0,atol=1e-10)
    contrasts=compare(cells,c)
    effects=dict(scope="FULL-SYSTEM INDEPENDENT EXACT PACKAGE EXTERNAL DEV",orders=SEEDS,
                 before_labels=signed,people=people,contrasts=contrasts,
                 absolute_pct=dict(zip(ARMS,cells.mean((1,2,3)).tolist(),strict=True)),
                 stock_label_repeat_changes=int(np.count_nonzero(bank[0]!=bank[1])),
                 hard_vs_stocks=[int(np.count_nonzero(bank[2]!=bank[i])) for i in range(2)], variation=variation)
    print("PASS exact package score",seal(Path("result.json"),effects),contrasts,flush=True)


if __name__=="__main__":
    run()
