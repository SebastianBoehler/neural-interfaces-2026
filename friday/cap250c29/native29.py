"""Execute genuine independent CSP fits on the complete live native upstream."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

import replay
from files29 import ACTOR_SHA,EXTERNAL_ARMS,prior,sha,stamp,write
from geometry29 import matched
from matched import capture
from native21 import stream
from scope import external,known
from scope29 import SEEDS,schedule
from worker import authentic_stock


def run(template,p,eeg,rows):
    begin=perf_counter()
    tape,stock,protected=capture(template,eeg,rows)
    x=np.stack([v[0] for v in tape]); rec=np.asarray([v[7] for v in tape],dtype=bool)
    s=protected["stop"]
    assert (s is None and not rec.any()) or int(np.flatnonzero(rec)[0])==s
    assert not np.any(rec[:-1]&~rec[1:])
    scores,work=matched(template.cp,p,x,stock[:,0],stock[:,1],rec,stock[:,2])
    assert not template.cp.t and not template.v5hist
    return dict(rows=rows.tolist(),labels=dict(zip(EXTERNAL_ARMS,scores.argmax(2).tolist(),strict=True)),
                work=work,protected=protected,raw=stamp(stock[:,0]),count=stamp(stock[:,1]),
                outputs=[stamp(a) for a in scores],seconds=perf_counter()-begin), (x,stock,rec,scores)


def smoke(template,p,eeg,rows,anchor,channels):
    a,frames=run(template,p,eeg,rows)
    assert a["labels"]["stock"]==anchor["labels"] and a["protected"]["activity"]==anchor["activity"]
    assert all(s["refits"]==5 and s["fusions"]==30 for s in a["work"].values())
    x,r,labels=known(channels)
    known_row,_=run(template,p,x,r)
    assert known_row["labels"]["stock"]==labels and len(set(known_row["outputs"]))==1
    assert all(r["stop"]==10 and r["fusions"]==r["refits"]==0 for r in known_row["work"].values())
    x,stock,rec,expected=frames
    for n in (31,105):
        pfx,_=matched(template.cp,p,x[:n],stock[:n,0],stock[:n,1],rec[:n],stock[:n,2])
        np.testing.assert_array_equal(pfx,expected[:,:n])
    return dict(unrecognized=a,recognized=known_row,prefix=[31,105])


def main(mode,index):
    assert torch.__version__=="2.14.1+cu130" and "RTX 2080 Ti" in torch.cuda.get_device_name()
    assert sha("/package/submission.py")==ACTOR_SHA and "neural-interfaces-2026/.venv/" not in Path("/proc/self/maps").read_text()
    index=5 if mode=="smoke" else index
    p,info,digest=prior("ALL")
    assert info["source_rows"]==2655 and not info["scherer_access"]
    eeg,who,channels,old,anchor,archive=external(index)
    template=replay.load(Path("/package"),channels)
    assert template.routed==1 and template.r15 and not template.v6r and template.ft.cap==250
    rows=old[0][who+"|1"]
    cold=stream(template,eeg,rows,"stock")
    authentic_stock(cold,anchor)
    record=dict(cold=cold[0],cold_rows=rows.tolist())
    if mode=="smoke":
        record.update(smoke(template,p,eeg,rows,anchor,channels))
        name="smoke.json"
    else:
        witness=json.loads(Path("smoke.json").read_text())
        assert witness["passed"] and witness["source"]==sha("source.sha256") and witness["prior"]==digest
        assert not witness["held_labels_opened"]
        with np.load("/external.npz",allow_pickle=False) as z:
            plans=schedule(z["subject"],z["session"])
        runs={}
        for seed in SEEDS:
            for visit in (1,2):
                key=f"{who}|{visit}"
                row,_=run(template,p,eeg,plans[seed][key])
                assert row["protected"]["stop"] is None and row["work"]["transport"]["fusions"]>0
                runs[f"{key}|{seed}"]=row
                print("SEALED F-C29",key,seed,row["work"]["transport"]["score_move"],flush=True)
        record["sessions"]=runs
        name=f"person_{index}.json"
    digest=write(name,dict(passed=True,source=sha("source.sha256"),scope=mode,index=index,person=who,
                prior=digest,data_source=sha("source.json"),archive=archive,device=torch.cuda.get_device_name(),
                held_labels_opened=False,**record))
    print("PASS F-C29 native",mode,index,digest,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("smoke","produce"))
    p.add_argument("index",type=int,choices=range(9))
    a=p.parse_args()
    main(a.mode,a.index)
