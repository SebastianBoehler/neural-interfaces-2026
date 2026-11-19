"""Pair the fixed causal emission policy on sealed complete native paths."""

import json
from pathlib import Path

import numpy as np

from scope import SEEDS, schedule
from score import DATA, changes, official_cells, seal
from score_shared21 import full_bank
from settings import sha, stamp


def select(bank,subject,day):
    path=Path("policy.json")
    assert not path.stat().st_mode&0o222
    digest=sha(path)
    assert digest=="e635e415c35ff410365e824c9b84a91f9eb613e3551e6e00ef44de10acba8024"
    policy=json.loads(path.read_text())
    assert policy["age_zero_based"] and tuple(policy["seeds"])==SEEDS
    assert policy["stock_after_real_recognition"] and policy["trajectories_start_from_session_age_zero"]
    plans=schedule(subject,day)
    age=np.full(bank.shape[1:],-1,dtype=np.int16)
    for j,seed in enumerate(SEEDS):
        for order in plans[seed].values():
            assert (age[j,order]==-1).all()
            age[j,order]=np.arange(len(order))
    assert (age>=0).all()
    # The parent authenticator requires actual `recognized == -1`, then verifies no RouteB or CSP stop.
    active=age>=policy["first_fbl2_age"]
    selected=np.where(active,bank[1],bank[0])
    assert active.sum()==3540 and (~active).sum()==720
    assert np.array_equal(selected[active],bank[1][active]) and np.array_equal(selected[~active],bank[0][~active])
    return selected,active,digest


def contrast(diff,resample):
    orders=diff.mean(axis=(0,1)).tolist()
    low,high=np.percentile(diff[resample].mean(axis=(1,2,3)),[10,90]).tolist()
    return dict(pp=float(diff.mean()),paired80=[low,high],orders_pp=dict(zip(SEEDS,orders,strict=True)))


def region(benchmark,trial,truth,mask):
    return dict(rows=int(mask.sum()),changed=int(((trial!=benchmark)&mask).sum()),
                rescue=int(((trial==truth)&(benchmark!=truth)&mask).sum()),
                loss=int(((trial!=truth)&(benchmark==truth)&mask).sum()))


def main():
    bank,subject,day,meta,people,proof,parent=full_bank()
    selected,active,policy=select(bank,subject,day)
    lock=seal(Path("selected_before_labels.json"),dict(policy=policy,parent=parent,selected=stamp(selected),
              mask=stamp(active),n_active=int(active.sum()),held_labels_opened=False))
    with np.load(DATA,allow_pickle=False) as z:
        y=z["y"]
    base,_=official_cells(bank,y,subject,day,meta,people)
    gate,_=official_cells(np.stack((bank[0],selected)),y,subject,day,meta,people)
    p=np.random.default_rng(468)
    samples=(p.integers(9,size=(10000,9,1,1)),p.integers(2,size=(10000,9,2,1)),p.integers(2,size=(10000,9,2,2)))
    effect=gate[1]-base[0]
    fbl=base[1]-base[0]
    to_stock=contrast(effect,samples)
    to_fbl=contrast(gate[1]-base[1],samples)
    stock_by_order,fbl_by_order=(d.mean(axis=(0,1)) for d in (effect,fbl))
    retain=[float(x/y) if y>0 else None for x,y in zip(stock_by_order,fbl_by_order,strict=True)]
    advance=to_stock["paired80"][0]>0 and min(stock_by_order)>0 and all(x is not None and x>=.6 for x in retain)
    result=dict(scope="External fixed emission selector on reused C21s people and deliveries",
                verdict="supported" if advance else "refuted" if to_stock["paired80"][1]<=0 else "undecided",
                seal=lock,policy=policy,versus_stock=to_stock,versus_fbl2=to_fbl,retain=retain,
                stock_pct=float(base[0].mean()),fbl2_pct=float(base[1].mean()),selected_pct=float(gate[1].mean()),
                overall_vs_stock=changes(np.stack((bank[0],selected)),y,np.ones(len(y),dtype=bool)),
                overall_vs_fbl2=changes(np.stack((bank[1],selected)),y,np.ones(len(y),dtype=bool)),
                protected_vs_fbl2=region(bank[1],selected,y,~active),eligible_vs_stock=region(bank[0],selected,y,active),
                people_pp=dict(zip(people,effect.mean(axis=(1,2)).tolist(),strict=True)),native_work=proof)
    digest=seal(Path("result.json"),result)
    print("PASS F-C21g external",digest,to_stock,to_fbl,retain,result["verdict"],flush=True)


if __name__=="__main__":
    main()
