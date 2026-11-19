"""Check the sticky retirement on real RTX streams and time paired warm execution."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import replay
from benchmark_utils.streaming import fresh_model
from fbl2 import FBL2Hook
from gate_rtx import equal_prefix, g_prefix, live
from live_gate import RetiringSelector
from matched import private
from mirror import Outer
from native21 import stream
from route_probe import RouteProbe
from scope import external, known
from settings import sha
from worker import authentic_stock


def proof(template, run, eeg, rows, expected, age, count):
    info, g = run
    stock, prefix, selected = g.stages()
    np.testing.assert_array_equal(stock[:,2].argmax(1), expected)
    np.testing.assert_array_equal(np.stack([e[0] for e in g.events]), eeg[rows].numpy())
    control = fresh_model(template)
    old_branch = Outer(control)
    old = private(old_branch, g.events, [e[2] for e in g.events])
    np.testing.assert_array_equal(stock, old)
    hook = FBL2Hook(control, "fbl2")
    candidate = Outer(control)
    route = RouteProbe(SimpleNamespace(routeB=candidate.route))
    alternative = private(candidate, g.events, [hook.project(e[2], e[-2], e[-1]) for e in g.events])
    n = len(rows) if age < 0 else age
    np.testing.assert_array_equal(prefix, alternative[:n])
    use = (np.arange(len(rows)) >= 20) & ~np.array([e[7] for e in g.events])
    wanted = np.where(use[:,None], alternative[:,2], old[:,2])
    np.testing.assert_array_equal(selected, wanted)
    np.testing.assert_array_equal(wanted.argmax(1), info["labels"])
    b = info["alternate"]
    if ((b["recognized"], b["eligible"], b["alternate_rows"], b["retired_at"])
            != (age, count, n, None if age < 0 else age)
            or not np.array_equal(g.outer.fit, [i for i in candidate.fit if i < n])
            or b["hook"]["calls"] != n or b["hook"]["moved"] != n
            or b["hook"]["warm_rows"] != sum("fbl" in e[-1] for e in g.events[:n])):
        raise ValueError("causal retirement state, actual alternate coverage or live activity differs")
    return dict(info, exact_dual_emission=True, exact_prelatch_state=True, route=route.summary(), rows=rows.tolist(),
                selected_changes=int(np.count_nonzero(wanted.argmax(1) != old[:,2].argmax(1))))


def smoke(template, eeg, who, orders, channels, anchor):
    rows = orders[0][who+"|1"]
    cold = stream(template,eeg, rows, "stock")
    authentic_stock(cold, anchor)
    full = live(template,eeg,rows,True,RetiringSelector)
    g = proof(template,full,eeg,rows,anchor["labels"],-1,100)
    authentic_stock((dict(g, labels=anchor["labels"]), None), anchor)
    w=g["alternate"]
    if (w["alternate_refits"],w["alternate_fused"],w["hook"]["warm_rows"]) != (5,30,100):
        raise ValueError("unrecognized G alternate work changed")
    px,r,y=known(channels)
    true_known = live(template,px,r,True,RetiringSelector)
    p = proof(template,true_known,px,r,y,10,0)
    if p["route"]["routed"]!=21 or p["selected_changes"]!=0:
        raise ValueError("the real known owner did not protect the whole session")
    live(template,eeg,orders[0][who+"|2"][:17],selector=RetiringSelector)
    rep = stream(template,eeg,rows[:31],"stock")
    pref = g_prefix(template,cold,full[1],live(template,eeg,rows[:31],True,RetiringSelector)[1],rep)
    equal_prefix(true_known[1],live(template,px,r[:17],True,RetiringSelector)[1],17)
    return dict(unrecognized=g,known=p,unrecognized_prefix=pref,recognized_prefix_exact=True)


def timing(template,eeg,who,orders,channels,index):
    report=json.loads(Path("smoke.json").read_text())
    late=json.loads(Path("delayed.json").read_text())
    if (any(x["source"] != sha("source.sha256") or x["held_labels_opened"] or not x["passed"] for x in (report,late))
            or not late["exact_prelatch_state"] or late["accepted"]!=list(range(20,28))):
        raise ValueError("timing requires both authentic retirement checks")
    if index==0:
        x,rows=eeg,orders[0][who+"|1"]
        key,order="unrecognized",("stock","retired","retired","stock")
    else:
        x,rows,_=known(channels)
        key,order="known",("stock","full","retired","retired","full","stock")
    want=report[key]
    warm=stream(template,x,rows,"stock",inspect=False)[0]
    runs=[]
    for arm in order:
        if arm=="stock":
            r=stream(template,x,rows,"stock",inspect=False)[0]
        elif arm=="full":
            r=live(template,x,rows)[0]
        else:
            r=live(template,x,rows,selector=RetiringSelector)[0]
        names=("rows","recognized","eligible","alternate_rows","alternate_refits","alternate_fused","retired_at")
        if (r["activity"]!=warm["activity"] or r["activity"]!=want["activity"]
                or r["labels"]!=(warm["labels"] if arm=="stock" else want["labels"])
                or arm=="retired" and any(r["alternate"][k]!=want["alternate"][k] for k in names)
                or arm=="full" and r["alternate"]["alternate_rows"]!=len(rows)):
            raise ValueError("timed native retirement changed labels or measured activity")
        runs.append(r)
    stamp={k:np.asarray([r["seconds"] for r, role in zip(runs,order,strict=True) if role==k]) for k in dict.fromkeys(order)}
    pair=stamp["retired"]/stamp["stock"]
    result=dict(fixture=key,order=order,runs=runs,ratio=float(stamp["retired"].mean()/stamp["stock"].mean()),
                paired=pair.tolist(),worst=float(pair.max()),stock_warm=warm)
    if index==1:
        result["retired_vs_full"] = float(stamp["retired"].mean()/stamp["full"].mean())
    return result


def main(mode,index):
    if torch.__version__!="2.14.1+cu130" or "RTX 2080 Ti" not in torch.cuda.get_device_name():
        raise ValueError("exact RTX worker required")
    eeg,who,channels,orders,anchor,archive=external(5)
    template=replay.load(Path("/package"),channels)
    if mode=="smoke":
        payload=smoke(template,eeg,who,orders,channels,anchor)
    else:
        payload=timing(template,eeg,who,orders,channels,index)
    p=Path("smoke.json" if mode=="smoke" else f"timing_{index}.json")
    with p.open("x") as f:
        json.dump(dict(passed=True,source=sha("source.sha256"),held_labels_opened=False,archive=archive,
                       device=torch.cuda.get_device_name(),**payload),f,sort_keys=True,allow_nan=False)
        f.write("\n")
    p.chmod(0o444)
    print("PASS F-C21gt",mode,index,sha(p),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("smoke","timing"))
    p.add_argument("index",type=int,choices=(0,1))
    args=p.parse_args()
    main(args.mode,args.index)
