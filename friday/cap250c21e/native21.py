"""Run independent stock, exact no-op and FBLight double in the authentic Solver."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

import replay
from adapter import Adapter as UpstreamAudit
from benchmark_utils.streaming import fresh_model
from fbl2 import FBL2Hook
from route_probe import RouteProbe
from scope import ARMS, SEEDS, external, known
from settings import sha
from worker import authentic_stock, close, same_upstream


def stream(template,eeg,rows,arm,*,inspect=True):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start=perf_counter()
    model=fresh_model(template)
    assert not model.v5hist and not model.cp.t and not model.dead
    hook=FBL2Hook(model,arm) if inspect or arm=="fbl2" else None
    stage, order = [], []
    if inspect:
        audit=UpstreamAudit(model,"stock",())
        consuming=RouteProbe(model)
        cp=model.cp.step

        def terminal(x,p,count,recognized):
            value=cp(x,p,count,recognized)
            stage.append(np.stack((p,count,value)))
            return value
        model.cp.step=terminal
    loaded=perf_counter()
    labels,times=[],[]
    for r in rows:
        t=perf_counter()
        out=model.predict(eeg[int(r):int(r)+1].to("cuda:0"))
        assert out.shape==(1,) and out.dtype==torch.int64 and out.device.type=="cpu"
        if inspect:
            hook.check(model.trace["v6"])
            assert model.trace["v6"]["mix6"] is consuming.last
            audit.record()
        assert not any(p.dead for p in (model,model.cf,model.cp))
        labels.append(int(out[0]))
        times.append(perf_counter()-t)
    torch.cuda.synchronize()
    end=perf_counter()
    activity=replay.activity(model)
    assert activity["windows"]==len(rows) and activity["errors"]==activity["ft_over"]==0
    assert activity["ft_steps"]==20*activity["ft_refits"]
    assert model.cp.stopped==(activity["recognized"] if activity["recognized"]>=0 else None)
    info=dict(labels=labels,activity=activity,seconds=end-start,restore_s=loaded-start,
              first_s=times[0],p95_s=float(np.percentile(times,95)),max_s=max(times),
              peak_reserved=torch.cuda.max_memory_reserved())
    output={}
    if inspect:
        assert len(audit.refits)==activity["ft_refits"]
        assert hook.calls==len(rows)==len(consuming.errors)
        assert hook.warm==model.log6["ft"]
        if arm=="fbl2":
            assert hook.moved==len(rows) and hook.mix_delta>.003
        if arm=="identity":
            assert hook.moved==0 and hook.mix_delta==0.
        order=consuming.summary()
        info.update(hook=hook.summary(),route=order)
        output=dict(upstream=np.asarray(audit.trace),loss=audit.refits,terminal=np.asarray(stage)[:,2],
                    stage=np.asarray(stage))
    return info,output


def pair(stock,trial):
    proof=same_upstream(stock,trial)
    assert stock[0]["route"]["gates"]==trial[0]["route"]["gates"]
    return dict(proof, exact_gate=True)


def smoke(template,eeg,channels,who,orders,anchor):
    assert who=="Scherer2015/G"
    rows=orders[0][who+"|1"]
    cold=stream(template,eeg,rows,"stock")
    authentic_stock(cold,anchor)
    stock=stream(template,eeg,rows,"stock")
    authentic_stock(stock,anchor)
    off=stream(template,eeg,rows,"identity")
    authentic_stock(off,anchor)
    identity=pair(stock,off)
    identity["stage_error"]=close(stock[1]["stage"],off[1]["stage"])
    trial=stream(template,eeg,rows,"fbl2")
    active=pair(stock,trial)
    assert trial[0]["hook"]["warm_rows"]>0 and trial[0]["hook"]["fbl_warm_delta"]>.01
    assert trial[0]["activity"]["cp_fused"]>0 and not trial[0]["route"]["routed"]
    stream(template,eeg,orders[0][who+"|2"][:17],"fbl2")
    short=stream(template,eeg,rows[:31],"fbl2")
    prefix=close(trial[1]["stage"][:31],short[1]["stage"])
    assert trial[0]["labels"][:31]==short[0]["labels"]
    x,p,wanted=known(channels)
    baseline=stream(template,x,p,"stock")
    native=stream(template,x,p,"fbl2")
    recognized=pair(baseline,native)
    assert baseline[0]["labels"]==wanted
    assert native[0]["activity"]["recognized"]==10 and native[0]["route"]["routed"]>0
    native_share=baseline[0]["hook"]["route_share"]
    assert native[0]["hook"]["original_share"]==native_share
    assert abs(native_share-1/16)<=1e-16 and native[0]["hook"]["route_share"]==1/17
    assert native[0]["hook"]["frozen_rows"]==len(p) and not native[0]["hook"]["warm_rows"]
    return dict(cold=cold[0],stock=stock[0],identity=off[0],fbl2=trial[0],identity_proof=identity,
                protected=active,prefix_error=prefix,recognized=dict(stock=baseline[0],fbl2=native[0],proof=recognized))


def produce(template,eeg,who,orders,anchor,index):
    canonical=orders[0][who+"|1"]
    cold=stream(template,eeg,canonical,"stock")
    authentic_stock(cold,anchor)
    for arm in ARMS:
        stream(template,eeg,canonical[:17],arm)
    results={}
    for j,seed in enumerate(SEEDS):
        for i,day in enumerate((1,2)):
            key=f"{who}|{day}"
            rows=orders[seed][key]
            seq=ARMS if (index+j+i)%2==0 else ARMS[::-1]
            runs={a:stream(template,eeg,rows,a) for a in seq}
            proof=pair(runs["stock"], runs["fbl2"])
            b=runs["fbl2"][0]
            assert b["activity"]["recognized"]==-1 and b["activity"]["cp_fused"]>0 and b["hook"]["warm_rows"]>0
            results[f"{key}|{seed}"]=dict(rows=rows.tolist(),order=list(seq),proof=proof,
                                         runs={a:v[0] for a,v in runs.items()})
            print("SEALED F-C21e",key,seed,proof["max_upstream"],flush=True)
    return dict(cold=cold[0],cold_rows=canonical.tolist(),results=results)


def timings(template,eeg,who,channels,orders,index):
    if index==0:
        x,rows=eeg,orders[0][who+"|1"]
    else:
        x,rows,_=known(channels)
    seq=("stock","fbl2","fbl2","stock") if index==0 else ("fbl2","stock","stock","fbl2")
    warm=stream(template,x,rows,"stock",inspect=False)[0]
    repeats=[stream(template,x,rows,a,inspect=False)[0] for a in seq]
    stock=[r for a,r in zip(seq,repeats,strict=True) if a=="stock"]
    boosted=[r for a,r in zip(seq,repeats,strict=True) if a=="fbl2"]
    assert stock[0]["activity"]==stock[1]["activity"]==warm["activity"]
    exclude={"cp_refits","cp_fused"}
    for b in boosted:
        assert all(b["activity"][k]==stock[0]["activity"][k] for k in b["activity"] if k not in exclude)
    a,b=(np.array([x["seconds"] for x in arm]) for arm in (stock,boosted))
    ratio=float(b.mean()/a.mean())
    return dict(warm=warm,paired=(b/a).tolist(),ratio=ratio,worst=float((b/a).max()),rounds=list(seq),
                runs=repeats,fixture="Scherer/G day 1" if index==0 else "PROTEUS/P10 visit 3")


def main(mode,index):
    assert torch.__version__=="2.14.1+cu130" and "RTX 2080 Ti" in torch.cuda.get_device_name()
    assert "neural-interfaces-2026/.venv/" not in Path("/proc/self/maps").read_text()
    source=sha("source.sha256")
    if mode!="smoke":
        witness=json.loads(Path("smoke.json").read_text())
        assert witness["source"]==source and witness["passed"] and not witness["held_labels_opened"]
    person=index if mode=="produce" else 5
    eeg,who,channels,orders,reference,archive=external(person)
    template=replay.load(Path("/package"),channels)
    assert template.routed==1 and template.r15 and not template.v6r and template.ft.cap==250 and template.perm is None
    if mode=="smoke":
        outcome=smoke(template,eeg,channels,who,orders,reference)
        name="smoke"
    elif mode=="produce":
        outcome=produce(template,eeg,who,orders,reference,person)
        name=f"person_{person}"
    else:
        assert index in (0,1)
        outcome=timings(template,eeg,who,channels,orders,index)
        name=f"timing_{index}"
    assert not template.v5hist and template.cp.t==0 and not template.dead
    path=Path(name+".json")
    with path.open("x") as f:
        json.dump(dict(passed=True,source=source,scope=mode,index=index,person=who,archive=archive,
                       device=torch.cuda.get_device_name(),held_labels_opened=False,**outcome),f,
                  sort_keys=True,allow_nan=False)
        f.write("\n")
    path.chmod(0o444)
    print("PASS F-C21e",mode,index,sha(path),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("smoke","produce","timing"))
    p.add_argument("index",type=int,choices=range(9))
    a=p.parse_args()
    main(a.mode,a.index)
