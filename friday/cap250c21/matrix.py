"""Seal the corrected stock and three actual outer contrasts with native tails."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from bridge_orders import write_npz, write_receipt
from bridge_owner import MEANS, sha, stamp, start_module
from cc30 import contents
from complete_zero import FEAT
from factor import TREATMENTS, variants


WORK=Path(__file__).resolve().parent
TAIL=MEANS/"fc20_outer_v1/worker.py"
run_tail=start_module("fc20_locked_tail",TAIL,TAIL.read_text()).run_tail
ARMS=("stock",*TREATMENTS)


def main(order,mode):
    start=perf_counter()
    smoke=mode=="smoke"
    revision=sha(WORK/"experiment.sha256")
    if not smoke:
        checked=json.loads((WORK/"smoke.json").read_text())
        assert checked["passed"] and checked["source"]==revision and not checked["held_labels_opened"]
    cc,source=contents(order)
    with np.load(FEAT,allow_pickle=False) as z:
        people,visits=z["subject"],z["session"]
    with variants(order,cc) as ctx:
        count=ctx["count"]
        all_sessions=tuple(count.SESS)
        assert len(all_sessions)==14 and stamp(ctx["delivery"])==source["pinned_order"]
        rows={s:count.C.order_rows(order,s,count.PROTO) for s in all_sessions}
        p01=next(s for s,r in rows.items() if set(people[r])=={"Dreyer2026Proteus/P01"} and set(visits[r])=={"1"})
        assert len(rows[p01])==190
        latch=count.latch(order)
        peer=next(s for s in all_sessions if s!=p01 and latch[s] is not None and latch[s]<120)
        prefix=(p01,peer,p01)
        sessions=(p01,peer) if smoke else all_sessions
        delivered=np.concatenate([rows[s] for s in sessions])
        if not smoke:
            np.testing.assert_array_equal(delivered,ctx["delivery"])
        stages,telemetry={},{}
        for arm in ARMS:
            result,proof=run_tail(ctx,order,ctx["raw"][arm],sessions,prefix)
            stages[arm],telemetry[arm]=result,proof
            assert proof["count_effect"]>0 and proof["csp_refits"]>0
            if not smoke:
                assert (proof["csp_refits"],proof["csp_fusions"])==(126,1395)
            print("PASS F-C21 live tails",order,arm,proof["count_effect"],proof["csp_refits"],flush=True)
        for k in stages["stock"]:
            np.testing.assert_array_equal(stages["stock"][k][delivered],ctx["reference"][k][delivered])
            np.testing.assert_array_equal(stages["identity"][k][delivered],stages["stock"][k][delivered])
        for arm in ("SS","FBL2","JS"):
            changes=np.abs(stages[arm]["terminal"][delivered]-stages["stock"]["terminal"][delivered])
            assert changes.max()>1e-4
            assert any(telemetry[arm]["csp"][str(s)]["target_sha"] != telemetry["stock"]["csp"][str(s)]["target_sha"]
                       for s in sessions)
        summary=dict(passed=True,stage=mode,order=order,rows=len(delivered),sessions=len(sessions),source=revision,
                     anchor=ctx["parent"],zero=ctx["zero"],cc30=source,stock_bitwise=True,native_identity=True,
                     shadow_fixed=True,strict_past=True,held_labels_opened=False,
                     prefix=[str(s) for s in prefix],mixer=ctx["stages"],work=telemetry,seconds=perf_counter()-start)
        if not smoke:
            arrays=dict(index=np.arange(2655),delivery=delivered)
            arrays.update({f"{arm}|{k}":v for arm,source in stages.items() for k,v in source.items()})
            summary["output"]=write_npz(WORK/f"{order}.npz",arrays)
        path=WORK/f"{'smoke' if smoke else order}.json"
        write_receipt(path,**summary)
    print("PASS F-C21",order,mode,sha(path),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("order",choices=("full0","within0"))
    p.add_argument("mode",choices=("smoke","complete"))
    a=p.parse_args()
    assert a.mode!="smoke" or a.order=="full0"
    main(a.order,a.mode)
