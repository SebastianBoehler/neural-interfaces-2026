"""Run the exact cap250 upstream and actual matched CSP actors."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

import replay
from matched import capture
from native21 import stream
from readout import ARMS, evaluate, stamp
from scope import external, known
from scope24 import SEEDS, schedule
from settings import sha
from worker import authentic_stock


def session(template,eeg,rows):
    t=perf_counter()
    tape,stock,prior=capture(template,eeg,rows)
    upstream=perf_counter()-t
    x=np.stack([v[0] for v in tape])
    rec=np.asarray([v[7] for v in tape],dtype=bool)
    assert (not rec.any() and prior["stop"] is None) or np.flatnonzero(rec)[0]==prior["stop"]
    assert not np.any(rec[:-1] & ~rec[1:])
    t=perf_counter()
    scores,result=evaluate(template.cp,x,stock[:,0],stock[:,1],rec,stock[:,2])
    offsets=perf_counter()-t
    assert not template.cp.t and not template.v5hist
    return dict(rows=rows.tolist(),labels=dict(zip(ARMS,scores.argmax(2).tolist(),strict=True)),prior=prior,
                raw=stamp(stock[:,0]),count=stamp(stock[:,1]),score_hash=[stamp(s) for s in scores],arms=result,
                measured_upstream_s=upstream,matched_csp_s=offsets), (x,stock,rec,scores)


def smoke(template,eeg,rows,anchor,channels):
    g,events=session(template,eeg,rows)
    assert g["labels"]["stock"]==anchor["labels"] and g["prior"]["activity"]==anchor["activity"]
    assert g["arms"]["temporal"]["extra_usage"]==30==g["arms"]["sham"]["extra_usage"]
    for mode in ("temporal","sham"):
        p=g["arms"][mode]
        assert p["max_q_change"]>.05 and p["max_score_change"]>1e-4
    x,ix,target=known(channels)
    other,_=session(template,x,ix)
    assert other["labels"]["stock"]==target and other["prior"]["stop"]==10
    assert all(a["fusions"]==0 and a["features"]==10 and a["max_score_change"]==0. for a in other["arms"].values())
    x,stock,rec,full=events
    for n in (31,105):
        prefix,_=evaluate(template.cp,x[:n],stock[:n,0],stock[:n,1],rec[:n],stock[:n,2])
        np.testing.assert_array_equal(prefix,full[:,:n])
    return dict(unrecognized=g,recognized=other,prefix=[31,105])


def main(mode,index):
    assert torch.__version__=="2.14.1+cu130" and "RTX 2080 Ti" in torch.cuda.get_device_name()
    assert "neural-interfaces-2026/.venv/" not in Path("/proc/self/maps").read_text()
    i=5 if mode=="smoke" else index
    eeg,who,channels,old,anchor,archive=external(i)
    template=replay.load(Path("/package"),channels)
    assert template.routed==1 and template.r15 and not template.v6r and template.ft.cap==250
    p0=old[0][who+"|1"]
    cold=stream(template,eeg,p0,"stock")
    authentic_stock(cold,anchor)
    record=dict(cold=cold[0],cold_rows=p0.tolist())
    if mode=="smoke":
        record.update(smoke(template,eeg,p0,anchor,channels))
    else:
        locked=json.loads(Path("smoke.json").read_text())
        assert locked["passed"] and locked["source"]==sha("source.sha256") and not locked["held_labels_opened"]
        with np.load("/external.npz",allow_pickle=False) as z:
            plans=schedule(z["subject"],z["session"])
        result={}
        for seed in SEEDS:
            for v in (1,2):
                k=f"{who}|{v}"
                r,_=session(template,eeg,plans[seed][k])
                assert r["prior"]["activity"]["recognized"]==-1
                assert r["arms"]["temporal"]["extra_usage"]>0 and r["arms"]["sham"]["extra_usage"]>0
                assert r["arms"]["temporal"]["max_score_change"]>1e-4
                result[f"{k}|{seed}"]=r
                print("SEALED F-C24",k,seed,r["arms"]["temporal"]["max_score_change"],flush=True)
        record["sessions"]=result
    path=Path("smoke.json" if mode=="smoke" else f"person_{i}.json")
    with path.open("x") as f:
        json.dump(dict(passed=True,source=sha("source.sha256"),index=i,person=who,archive=archive,scope=mode,
                       device=torch.cuda.get_device_name(),held_labels_opened=False,**record),f,sort_keys=True,allow_nan=False)
        f.write("\n")
    path.chmod(0o444)
    print("PASS F-C24",mode,i,sha(path),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("smoke","produce"))
    p.add_argument("index",type=int,choices=range(9))
    a=p.parse_args()
    main(a.mode,a.index)
