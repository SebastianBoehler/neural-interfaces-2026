"""Run peer/sham updates and the authentic complete cap250 consumer."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

import replay
from benchmark_utils.streaming import fresh_model
from optimizer26 import Refit
from route_probe import RouteProbe
from scope import external,known
from scope26 import ARMS,SEEDS,schedule
from settings import sha,stamp


ARCHIVE="37c666a8a63c043c290bd35c527c26c6cfaf028bbe55c746cb8c685dac9fb69f"
HISTORY=("age","labels","confidence","gen_before","gen_after","frozen","raw")
PROTECTED=("windows","recognized","ft_refits","ft_relabels","ft_steps","ft_capped","ft_over","errors")
MAX_REPEAT=5e-5


def stream(template,eeg,rows,arm):
    start=perf_counter()
    model=fresh_model(template)
    assert not model.v5hist and model.cp.t==0 and not model.dead
    adapt,route=Refit(model,arm),RouteProbe(model)
    stages=[]
    terminal=model.cp.step

    def observe(x,raw,count,recognized):
        out=terminal(x,raw,count,recognized)
        stages.append(np.stack((raw,count,out)))
        return out

    model.cp.step=observe
    labels,mix=[],[]
    for r in rows:
        y=model.predict(eeg[int(r):int(r)+1].to("cuda:0"))
        assert y.shape==(1,) and y.dtype==torch.int64 and y.device.type=="cpu"
        assert not any(v.dead for v in (model,model.cf,model.cp))
        assert model.trace["v6"]["mix6"] is route.last
        labels.append(int(y[0]))
        mix.append(route.last.copy())
        adapt.row()
    torch.cuda.synchronize()
    a=replay.activity(model)
    assert len(rows)==a["windows"] and a["ft_over"]==a["errors"]==0
    assert a["ft_steps"]==20*a["ft_refits"] and a["ft_refits"]==a["ft_relabels"]==len(adapt.events)
    assert model.cp.stopped==(a["recognized"] if a["recognized"]>=0 else None)
    assert [v["age"] for v in adapt.events]==list(range(20,20+10*a["ft_refits"],10))
    live=np.asarray(stages)
    np.testing.assert_array_equal(live[:,2].argmax(1),labels)
    assert not template.v5hist and template.cp.t==0
    result=dict(labels=labels,activity=a,seconds=perf_counter()-start,events=adapt.events,fit=adapt.summary(),
                gate=route.summary(),protected=stamp(np.asarray(adapt.protected)),mixture=stamp(np.asarray(mix)),
                terminal=stamp(live[:,2]))
    return result,dict(stages=live,warm=np.asarray(adapt.warm),mix=np.asarray(mix),prot=np.asarray(adapt.protected))


def paired(a,b):
    s,x=a
    v,y=b
    assert all(s["activity"][k]==v["activity"][k] for k in PROTECTED)
    assert s["gate"]["gates"]==v["gate"]["gates"] and s["protected"]==v["protected"]
    np.testing.assert_array_equal(x["prot"],y["prot"])
    for p,q in zip(s["events"],v["events"],strict=True):
        for k in HISTORY:
            assert p[k]==q[k],(p["age"],k)
    for k in x:
        np.testing.assert_array_equal(x[k][:20],y[k][:20])
    assert s["labels"][:20]==v["labels"][:20]
    return {k:float(np.abs(x[k]-y[k]).max()) for k in ("stages","mix","warm")}


def identity(a,b):
    delta=paired(a,b)
    assert a[0]["labels"]==b[0]["labels"] and a[0]["activity"]==b[0]["activity"]
    assert max(delta.values())<=MAX_REPEAT,delta
    assert b[0]["fit"]["consumes"]==4*b[0]["fit"]["refits"] and b[0]["fit"]["touched"]==0
    return delta


def cold_control(a,anchor):
    assert a[0]["labels"]==anchor["labels"] and a[0]["activity"]==anchor["activity"]


def smoke(template,eeg,rows,anchor,channels,cold):
    ref=stream(template,eeg,rows,"stock")
    cold_control(ref,anchor)
    sham_zero=stream(template,eeg,rows,"identity")
    one=identity(ref,sham_zero)
    runs={}
    diff={}
    for variant in ("peer","sham"):
        z=stream(template,eeg,rows,variant)
        diff[variant]=paired(ref,z)
        f=z[0]["fit"]
        assert f["active"]>=5 and f["family_difference"]>1e-4 and f["consumes"]==4*f["refits"]
        assert diff[variant]["warm"]>1e-5 and diff[variant]["stages"]>1e-4 and diff[variant]["mix"]>1e-5
        runs[variant]=z
    assert paired(runs["peer"],runs["sham"])["warm"]>1e-5
    x,r,target=known(channels)
    recognized={}
    for variant in ARMS:
        a=stream(template,x,r,variant)
        assert a[0]["labels"]==target and a[0]["activity"]["recognized"]==10
        assert a[0]["fit"]["consumes"]==0 and not a[0]["events"] and a[0]["gate"]["routed"]==21
        recognized[variant]=a[0]
    assert len({a["terminal"] for a in recognized.values()})==1
    prefix=stream(template,eeg,rows[:31],"peer")
    full=runs["peer"]
    equal=tuple(k for k in HISTORY)
    assert len(prefix[0]["events"])==2
    assert all(all(a[k]==b[k] for k in equal) for a,b in zip(full[0]["events"][:2],prefix[0]["events"],strict=True))
    delta=max(float(np.abs(prefix[1][k]-full[1][k][:31]).max()) for k in prefix[1])
    assert delta<=MAX_REPEAT and prefix[0]["labels"]==full[0]["labels"][:31]
    for k in prefix[1]:
        np.testing.assert_array_equal(prefix[1][k][:20],full[1][k][:20])
    return dict(cold=cold[0],stock=ref[0],identity=sham_zero[0],zero=one,recognized=recognized,
                arms={k:v[0] for k,v in runs.items()},treatment_deltas=diff,prefix=delta)


def main(mode,index):
    assert torch.__version__=="2.14.1+cu130" and "RTX 2080 Ti" in torch.cuda.get_device_name()
    assert sha("/package/submission.py")==ARCHIVE
    assert "neural-interfaces-2026/.venv/" not in Path("/proc/self/maps").read_text()
    index=5 if mode=="smoke" else index
    eeg,who,channels,prior,anchor,archive=external(index)
    with np.load("/external.npz",allow_pickle=False) as z:
        plans=schedule(z["subject"],z["session"])
    template=replay.load(Path("/package"),channels)
    p0=prior[0][who+"|1"]
    cold=stream(template,eeg,p0,"stock")
    cold_control(cold,anchor)
    if mode=="smoke":
        detail=smoke(template,eeg,p0,anchor,channels,cold)
        path=Path("smoke.json")
    else:
        witness=json.loads(Path("smoke.json").read_text())
        assert witness["passed"] and witness["source"]==sha("source.sha256") and not witness["held_labels_opened"]
        i=stream(template,eeg,p0[:31],"identity")
        s=stream(template,eeg,p0[:31],"stock")
        control=identity(s,i)
        assert i[0]["fit"]["consumes"]==8
        windows={}
        for j,seed in enumerate(SEEDS):
            for d in (1,2):
                k=f"{who}|{d}"
                rows=plans[seed][k]
                shift=(index+j+d)%3
                sequence=ARMS[shift:]+ARMS[:shift]
                arms={a:stream(template,eeg,rows,a) for a in sequence}
                evidence={a:paired(arms["stock"],arms[a]) for a in ("peer","sham")}
                for a in ("peer","sham"):
                    assert arms[a][0]["fit"]["active"]>=5 and arms[a][0]["activity"]["recognized"]==-1
                    assert evidence[a]["warm"]>1e-5 and evidence[a]["stages"]>1e-4
                    assert arms[a][0]["activity"]["cp_fused"]>0
                windows[f"{k}|{seed}"]=dict(rows=rows.tolist(),order=list(sequence),
                                            runs={a:v[0] for a,v in arms.items()},movement=evidence)
                print("SEALED F-C26",k,seed,evidence,flush=True)
        detail=dict(cold=cold[0],cold_rows=p0.tolist(),zero=control,zero_scope=i[0]["fit"],sessions=windows)
        path=Path(f"person_{index}.json")
    with path.open("x") as f:
        json.dump(dict(passed=True,index=index,device=torch.cuda.get_device_name(),source=sha("source.sha256"),
                       scope=mode,person=who,archive=archive,held_labels_opened=False,**detail),f,
                  sort_keys=True,allow_nan=False)
        f.write("\n")
    path.chmod(0o444)
    print("PASS F-C26",mode,index,sha(path),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("smoke","produce"))
    p.add_argument("index",type=int,choices=range(9))
    a=p.parse_args()
    main(a.mode,a.index)
