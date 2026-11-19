"""Prove exact current count-view reuse and warm paired RTX timing."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import replay
from combined_rtx import SIGNATURES, receipt
from combined_selector import SharedCovRetiringSelector
from count_state import verify_count_state
from gate_rtx import equal_prefix, g_prefix, live
from native21 import stream
from scope import external, known
from settings import sha
from shared_view import SharedViewRetiringSelector
from worker import authentic_stock


def verify(template, run, eeg, rows, wanted, age, eligible):
    proof = receipt(template, run, eeg, rows, wanted, age, eligible)
    expected = len(rows) if age < 0 else age
    m = proof["alternate"]
    if (m["count_view_calls"],m["verified_count_views"],m["shared_calls"]) != (expected,expected,expected):
        raise ValueError("the real native count view path did not run on every active frame")
    proof["count"] = verify_count_state(template,run[1])
    return proof


def smoke(template, eeg, who, channels, orders, anchor):
    rows = orders[0][who+"|1"]
    cold = stream(template,eeg,rows,"stock")
    authentic_stock(cold,anchor)
    g = live(template,eeg,rows,True,SharedViewRetiringSelector)
    one = verify(template,g,eeg,rows,anchor["labels"],-1,100)
    authentic_stock((dict(one,labels=anchor["labels"]),None),anchor)
    info = one["alternate"]
    if ((info["stock_count_latch"],info["alternate_count_latch"],info["alternate_refits"],
         info["alternate_fused"],info["hook"]["warm_rows"]) != (50,50,5,30,100)):
        raise ValueError("native natural G count, fitting or warm activity differs")
    x, rows_p10, expected = known(channels)
    p = live(template,x,rows_p10,True,SharedViewRetiringSelector)
    two = verify(template,p,x,rows_p10,expected,10,0)
    if two["route"]["routed"] != 21 or two["selected_changes"] != 0:
        raise ValueError("native recognized RTX fixture did not stay stock")
    live(template,eeg,orders[0][who+"|2"][:17],selector=SharedViewRetiringSelector)
    independent = stream(template,eeg,rows[:31],"stock")
    short = live(template,eeg,rows[:31],True,SharedViewRetiringSelector)[1]
    prefix = g_prefix(template,cold,g[1],short,independent)
    equal_prefix(p[1],live(template,x,rows_p10[:17],True,SharedViewRetiringSelector)[1],17)
    return dict(unrecognized=one,known=two,unrecognized_prefix=prefix,known_prefix_exact=True)


def timing(template, eeg, who, channels, orders, index):
    proof = json.loads(Path("smoke.json").read_text())
    delayed = json.loads(Path("delayed.json").read_text())
    if (any(x["source"] != sha("source.sha256") or x["held_labels_opened"] or not x["passed"] for x in (proof,delayed))
            or delayed["eligible_ages"] != list(range(20,28)) or not delayed["count"]["exact_alternate_count_state"]):
        raise ValueError("both pinned native count-state and retirement proofs are required")
    if index == 0:
        x, rows, name = eeg, orders[0][who+"|1"], "unrecognized"
    else:
        x, rows, _ = known(channels)
        name = "known"
    wanted = proof[name]
    warm = stream(template,x,rows,"stock",inspect=False)[0]
    order = ("stock","parent","reuse","reuse","parent","stock")
    runs = []
    for arm in order:
        if arm == "stock":
            r = stream(template,x,rows,"stock",inspect=False)[0]
        else:
            cls = SharedViewRetiringSelector if arm == "reuse" else SharedCovRetiringSelector
            r = live(template,x,rows,selector=cls)[0]
            if any(r["alternate"][key] != wanted["alternate"][key] for key in SIGNATURES):
                raise ValueError("native activity changed under timing")
            if arm == "reuse" and (r["alternate"]["count_view_calls"] != wanted["alternate"]["count_view_calls"]
                                   or r["alternate"]["verified_count_views"] != 0
                                   or r["alternate"]["alternate_count_latch"] != wanted["alternate"]["alternate_count_latch"]):
                raise ValueError("timed view sharing or native count latch changed")
        if r["activity"] != warm["activity"] or r["activity"] != wanted["activity"]:
            raise ValueError("original protected stock work differs")
        if r["labels"] != (warm["labels"] if arm == "stock" else wanted["labels"]):
            raise ValueError("timed worker changed a verified output")
        runs.append(r)
    t = {a:np.asarray([v["seconds"] for v,arm in zip(runs,order,strict=True) if arm==a]) for a in dict.fromkeys(order)}
    pair = t["reuse"]/t["stock"]
    ratio, worst = float(t["reuse"].mean()/t["stock"].mean()), float(pair.max())
    return dict(fixture=name,stock_warm=warm,runs=runs,order=order,ratio=ratio,paired=pair.tolist(),worst=worst,
                reuse_vs_parent=float(t["reuse"].mean()/t["parent"].mean()),meets_timing=ratio<=1.12 and worst<=1.15)


def main(mode,index):
    if torch.__version__ != "2.14.1+cu130" or "RTX 2080 Ti" not in torch.cuda.get_device_name():
        raise ValueError("pinned RTX worker required")
    eeg,who,channels,orders,anchor,archive = external(5)
    template = replay.load(Path("/package"),channels)
    result = smoke(template,eeg,who,channels,orders,anchor) if mode == "smoke" else timing(template,eeg,who,channels,orders,index)
    path = Path("smoke.json" if mode == "smoke" else f"timing_{index}.json")
    with path.open("x") as f:
        json.dump(dict(result,passed=True,source=sha("source.sha256"),archive=archive,
                       device=torch.cuda.get_device_name(),held_labels_opened=False),f,sort_keys=True,allow_nan=False)
        f.write("\n")
    path.chmod(0o444)
    print("PASS F-C21gcvw",mode,index,sha(path),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("smoke","timing"))
    p.add_argument("index",type=int,choices=(0,1))
    a=p.parse_args()
    main(a.mode,a.index)
