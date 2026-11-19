"""Replay only the modified terminal stage on the two signed corrected upstreams."""

import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from bridge_orders import write_npz, write_receipt
from bridge_owner import BASE, ROOT, pin, sha
from complete_tail import load_personalization
from complete_zero import FEAT, HASHES
from member_inputs import ORDERS, pinned_delivery
from readout import ARMS, evaluate, stamp


C19=BASE/"ni26_friday_causal_rebuild_20261118/fc19_csp_gate_v2"
PINS={
    "within0": ("5b3601bb18ca911a49daac9b8b8bf27d0894f1bdc2fd60570de9f6ff09ffef71",
                "950cda1f806543693ecd88f0029ca3e589ea44ff101a42bf493c47db2ac4559e",
                "f5f5a2f5ce8974038afa949e10e3368a3f5e38002831fdfb47373992570d3eb4"),
    "full0": ("c6e48aacfe076995f6f39e2acc5fe57086aa22177a217488d210e10f3f2b49bb",
              "794fc90d70f419d9ffe1563a179442c900b5528318b333e1fcb1c57b02d41162",
              "9b9dc4070b234a6df2c2d806c7ad5f2ddf2dbd231370fae688369a78b046663c"),
}


def inputs(order):
    old,data=C19/f"{order}.json",C19/f"{order}.npz"
    digest,bank,zero=PINS[order]
    pin(old,digest)
    pin(data,bank)
    pin(FEAT,HASHES[FEAT])
    proof=json.loads(old.read_text())
    assert not proof["held_labels_opened"] and proof["passed"] and proof["stock_bitwise"]
    assert proof["warm_steps"]==9600 and proof["strict_past"] and proof["no_recognition"]
    assert proof["archive_zero"]==zero and proof["output"]["sha256"]==bank and proof["order"]==order
    with np.load(FEAT,allow_pickle=False) as z:
        x=z["X"].astype(np.float64)
        sessions=np.char.add(np.char.add(z["subject"],"|"),z["session"])
    with np.load(data,allow_pickle=False) as z:
        np.testing.assert_array_equal(z["index"],np.arange(2655))
        delivery=z["delivery"]
        arrays=[z[f"stock|{key}"] for key in ("raw","filtered","scores")]
    exact=pinned_delivery(ROOT/"runs/shuffle_perms.npz")[ORDERS.index(order)]
    np.testing.assert_array_equal(delivery,exact)
    by_session={k:delivery[sessions[delivery]==k] for k in dict.fromkeys(sessions[delivery])}
    assert len(by_session)==14
    np.testing.assert_array_equal(np.concatenate(list(by_session.values())),delivery)
    return by_session,x,arrays,delivery


def main(order):
    start=perf_counter()
    source=sha(Path("source.sha256"))
    smoke=json.loads(Path("smoke.json").read_text())
    assert smoke["source"]==source and smoke["passed"] and not smoke["held_labels_opened"]
    sessions,x,stock,delivery=inputs(order)
    factory=load_personalization()
    cp=factory.CP.CausalPers(factory.CONFIG)
    p01=next(k for k in sessions if k.endswith("/P01|1"))
    first,other=sessions[p01],next(v for k,v in sessions.items() if k!=p01)
    rec=np.zeros(len(x),dtype=bool)

    def consume(idx):
        a,b,c=(u[idx] for u in stock)
        return evaluate(cp,x[idx],a,b,rec[idx],c)

    start_score,_=consume(first)
    consume(other)
    for n in (31,105):
        prefix,_=consume(first[:n])
        np.testing.assert_array_equal(prefix,start_score[:,:n])
    out=np.empty((len(ARMS),len(x),3),dtype=np.float64)
    work={}
    for k,idx in sessions.items():
        out[:,idx],info=consume(idx)
        s=info["stock"]
        t=info["temporal"]
        assert s["stop"] is None and s["fusions"]==len(idx)-90 and t["extra_usage"]>0
        assert t["max_score_change"]>1e-4
        work[k]=info
    np.testing.assert_array_equal(out[0],stock[2])
    np.testing.assert_array_equal(out[1],stock[2])
    report=dict(csp_refits=sum(x["stock"]["refits"] for x in work.values()),
                csp_fusions=sum(x["stock"]["fusions"] for x in work.values()),
                extra_readouts=sum(x["temporal"]["extra_usage"] for x in work.values()))
    assert (report["csp_refits"],report["csp_fusions"])==(126,1395)
    blob=write_npz(Path(f"public_{order}.npz").resolve(),dict(index=np.arange(2655),delivery=delivery,
                   scores=out,raw=stock[0],count=stock[1]))
    path=Path(f"public_{order}.json").resolve()
    write_receipt(path,passed=True,source=source,public_source=sha(Path("public.sha256")),held_labels_opened=False,
                  scope=order,stock_bitwise=True,identity_bitwise=True,prefix=[31,105],archived=PINS[order],
                  warm_steps=9600,no_recognition=True,bank=blob,raw=stamp(stock[0]),count=stamp(stock[1]),
                  work=report,match_proofs=work,seconds=perf_counter()-start)
    print("PASS F-C24 public",order,sha(path),report,flush=True)


if __name__=="__main__":
    assert len(sys.argv)==2 and sys.argv[1] in PINS
    main(sys.argv[1])
