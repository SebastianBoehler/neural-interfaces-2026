"""Transport fit-excluded source priors on both corrected stock and hard paths."""
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from bridge_orders import write_npz,write_receipt
from bridge_owner import BASE,pin,sha
from complete_tail import load_personalization
from complete_zero import FEAT
from files29 import EXTERNAL_ARMS,PEOPLE,PUBLIC_ARMS,prior,stamp
from geometry29 import matched
from public24 import PINS as C19,inputs


C20=BASE/"ni26_friday_causal_rebuild_20261118/fc20_outer_v1"
HARD={
    "within0":("a2249de5725ed36472ef6126ce1195b502fe1f3b9a47bef3124edf02bb81ebae",
               "73f14c4082174008aa43ab9b7447a48a85973eb12a2113b8723520177de9f3a5"),
    "full0":("1e3109313d14fb2d1a429762b1f2ef544bed3bb38759e8b1bbd0d0f12f8ac683",
             "1a5f0c2d9cb81e38cd8f11608440c237c032ec413efee4ad9675a9f7507b9a57"),
}


def hard_parent(order,rows,stock):
    receipt,bank=C20/f"{order}.json",C20/f"{order}.npz"
    pin(receipt,HARD[order][0]); pin(bank,HARD[order][1])
    check=json.loads(receipt.read_text())
    assert check["passed"] and check["stock_bitwise"] and check["native_identity"] and not check["held_labels_opened"]
    assert check["scope"]=="complete" and check["order"]==order and check["rows"]==2655
    assert check["shadow_fixed"] and check["new_csp_targets"]
    assert check["output"]["sha256"]==HARD[order][1]
    with np.load(bank,allow_pickle=False) as z:
        np.testing.assert_array_equal(z["delivery"],rows)
        np.testing.assert_array_equal(z["index"],np.arange(len(rows)))
        for v,k in zip(stock,("raw","filtered","terminal"),strict=True):
            np.testing.assert_array_equal(v,z[f"stock|{k}"])
        return [z[f"hard|{k}"] for k in ("raw","filtered","terminal")]


def main(order):
    start=perf_counter()
    current=sha(Path("source.sha256"))
    smoking=json.loads(Path("smoke.json").read_text())
    assert smoking["passed"] and smoking["source"]==current and not smoking["held_labels_opened"]
    assert smoking["data_source"]==sha(Path("source.json"))
    sessions,x,stock,order_rows=inputs(order)
    hard=hard_parent(order,order_rows,stock)
    factory=load_personalization()
    cp=factory.CP.CausalPers(factory.CONFIG)
    rec=np.zeros(len(x),dtype=bool)
    with np.load(FEAT,allow_pickle=False) as z:
        population=np.array([name.rsplit("/",1)[-1] for name in z["subject"]])
    matrices,pins={},{}
    for name in PEOPLE:
        p,m,pins[name]=prior(name)
        with np.load(f"prior_{name}.npz",allow_pickle=False) as z:
            np.testing.assert_array_equal(z["source_index"],np.flatnonzero(population!=name).astype("<i8"))
        assert m["source_rows"]==np.count_nonzero(population!=name) and name not in m["source_persons"]
        matrices[name]=p

    def play(rows,name):
        p=matrices[name]
        assert (population[rows]==name).all()
        a,b,c=(v[rows] for v in stock)
        capital,work=matched(cp,p,x[rows],a,b,rec[rows],c)
        a,b,c=(v[rows] for v in hard)
        wave,proof=matched(cp,p,x[rows],a,b,rec[rows],c,arms=("stock","transport"))
        return np.concatenate((capital,wave)),dict(work,hard=proof["stock"],transport_hard=proof["transport"])

    a=next(v for k,v in sessions.items() if k.endswith("/P01|1"))
    b=next(v for k,v in sessions.items() if not k.endswith("/P01|1"))
    check,_=play(a,"P01")
    play(b,population[b[0]])
    for n in (31,105):
        sample,_=play(a[:n],"P01")
        np.testing.assert_array_equal(sample,check[:,:n])
    out=np.empty((len(PUBLIC_ARMS),len(x),3),dtype=np.float64)
    work={}
    for session,rows in sessions.items():
        matrix,evidence=play(rows,population[rows[0]])
        assert tuple(evidence)==PUBLIC_ARMS
        out[:,rows]=matrix
        work[session]=evidence
    np.testing.assert_array_equal(out[0],stock[2]); np.testing.assert_array_equal(out[1],stock[2])
    np.testing.assert_array_equal(out[5],hard[2])
    for arm in PUBLIC_ARMS:
        assert sum(w[arm]["refits"] for w in work.values())==126
        assert sum(w[arm]["fusions"] for w in work.values())==1395
    bank=write_npz(Path(f"public_{order}.npz").resolve(),dict(index=np.arange(2655),delivery=order_rows,scores=out))
    result=Path(f"public_{order}.json").resolve()
    write_receipt(result,passed=True,source=current,public_source=sha(Path("public.sha256")),scope=order,
                  held_labels_opened=False,source_receipt=sha(Path("source.json")),bank=bank,folds=pins,
                  stock_parent=C19[order],hard_parent=HARD[order],stock_hash=stamp(out[0]),hard_hash=stamp(out[5]),
                  stock_zero=True,hard_zero=True,prefix=[31,105],no_recognition=True,
                  matched_work=work,seconds=perf_counter()-start)
    print("PASS F-C29 public",order,sha(result),flush=True)


if __name__=="__main__":
    assert len(sys.argv)==2 and sys.argv[1] in HARD
    main(sys.argv[1])
