"""Test direct hard ZIP same-input conformance and paired original RTX runtime."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import torch

import replay
from benchmark_utils.streaming import fresh_model
from build_hard import ACTORS,LOCK,STOCK
from fixture import chosen
from prove_mix import ObserveHard,measured,proof,protect
from settings import sha,stamp
from stock_drift import capture


ORACLE="d01c559bce326a7400fb983a06565aeae4acedd7d86502c6126da93c42370f22"
ORACLE_SMOKE="7521c87775762622eb6e851f3285d6781b3ba146547768899b7a56c0a192151a"
ARM_ORDER=("Scherer2015/G|1|0","P10-3")
TOL=5e-5


def fixtures():
    choices,(_,_,anchor,archive),channels,who,expected=chosen("smoke",5)
    assert who=="Scherer2015/G" and tuple(choices)==ARM_ORDER
    source=Path("/f32x")
    if any(sha(source/n)!=h for n,h in (("actors.json",LOCK),("pupils_smoke_5.npz",ORACLE),("smoke.json",ORACLE_SMOKE))):
        raise ValueError("original F32x fixture and actor identity differs")
    signed=json.loads((source/"smoke.json").read_text())
    if not signed["passed"] or signed["held_labels_opened"] or signed["actors"]!=LOCK or signed["pupil"]!=ORACLE:
        raise ValueError("expected authentic F32x blind fixture")
    values={}
    with np.load(source/"pupils_smoke_5.npz",allow_pickle=False) as bank:
        for key,(x,rows) in choices.items():
            np.testing.assert_array_equal(rows,bank[key+"|rows"])
            if stamp(x[rows].numpy()) != str(bank[key+"|x"]):
                raise ValueError("actual raw window bytes differ from the signed CPU oracle")
            np.testing.assert_array_equal(bank[key+"|p"][:,0],bank[key+"|p"][:,1])
            np.testing.assert_array_equal(expected[key],signed["results"][key]["stock"])
            values[key]=(x[rows].to("cuda:0"),rows,bank[key+"|p"][:,0],np.array(expected[key]),
                         np.array(signed["results"][key]["hard"]),signed["results"][key])
    if len(values[ARM_ORDER[0]][0])!=120 or len(values[ARM_ORDER[1]][0])!=31 or anchor["activity"]["ft_steps"]!=200:
        raise ValueError("genuine recognized or warm unrecognized scope differs")
    return values,channels,archive


def load(stock,trial,root,channels):
    expected=Path("archive.sha256").read_text().split()
    if sha(stock)!=STOCK or expected != [sha(trial),trial.name]:
        raise ValueError("frozen stock or signed trial bytes differ")
    with ZipFile(stock) as s,ZipFile(trial) as t:
        names=s.namelist()
        if len(names)!=39 or t.namelist()!=names+[f"hard_s1c_s{i}.pt" for i in range(2)]:
            raise ValueError("unexpected deterministic package manifest")
        for name in names:
            if name!="submission.py" and s.read(name)!=t.read(name):
                raise ValueError(f"inherited asset differs: {name}")
        import hashlib
        for i,wanted in enumerate(ACTORS):
            if hashlib.sha256(t.read(f"hard_s1c_s{i}.pt")).hexdigest()!=wanted:
                raise ValueError("packaged actor identity differs")
        s.extractall(root/"stock")
        t.extractall(root/"trial")
    a,b=(replay.load(root/name,channels) for name in ("stock","trial"))
    if type(a).__name__!="Ensemble6" or type(b).__name__!="HardWaveEnsemble":
        raise ValueError("official Solver factory did not activate direct hard packaging")
    return a,b


def resets(trial,peer,windows,q,ref):
    n=17
    start=fresh_model(trial)
    for x in windows[:n]:
        start.predict(x[None])
    result={}
    for mode, model in (("active_copy",fresh_model(start)),("reused",start)):
        model.reset_state()
        model.predict(peer)
        model.reset_state()
        if model.hard_actors is not None or model.hard_window is not None or model.cp.t or model.cf.t:
            raise ValueError("actor or native state crossed a session boundary")
        ObserveHard(model,q[:n])
        live=capture(model,windows[:n])
        np.testing.assert_array_equal(ref["labels"][:n],live["labels"])
        np.testing.assert_allclose(ref["stages"][:n],live["stages"],rtol=0,atol=TOL)
        result[mode]=dict(exact_cpu_actor_reset=True,labels_exact=True,native_rows=n,
                          fresh_stage_max=float(np.abs(ref["stages"][:n]-live["stages"]).max()))
    return result


def smoke(stock,trial,cases):
    results={}
    for i,key in enumerate(ARM_ORDER):
        x,rows,q,baseline,hard,signed=cases[key]
        a,b=measured(stock,x),measured(stock,x)
        for base in (a,b):
            np.testing.assert_array_equal(base[1]["labels"],baseline)
            if base[1]["work"]!=signed["activity"]:
                raise ValueError("original archived RTX stock work differs")
        changed=measured(trial,x,q)
        current=changed[1]
        np.testing.assert_array_equal(current["labels"],hard)
        native=proof(stock,current,changed[3])
        aa=protect(a,b)
        drift=protect(b,changed)
        if native["warm_rows"]!=(100 if i==0 else 0) or current["work"]["recognized"]!=(-1 if i==0 else 10):
            raise ValueError("the original warmed raw seed or true recognized route went inactive")
        peer=cases[ARM_ORDER[1-i]][0][:1]
        results[key]=dict(rows=rows.tolist(),stock=baseline.tolist(),hard=hard.tolist(),stock_work=b[1]["work"],
                          hard_work=current["work"],native=native,protected_aa=aa,protected_independent=drift,
                          resets=resets(trial,peer,x,q,current))
    return results


def timed_run(template,windows,changed):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    tick=perf_counter()
    model=fresh_model(template)
    calls=0
    if changed:
        original=model.hard_probability

        def call(raw):
            nonlocal calls
            calls+=1
            return original(raw)
        model.hard_probability=call
    restored=perf_counter()
    labels,times=[],[]
    for x in windows:
        begin=perf_counter()
        y=model.predict(x[None])
        times.append(perf_counter()-begin)
        if y.shape!=(1,) or y.dtype!=torch.long or y.device.type!="cpu":
            raise ValueError("timed official interface changed")
        labels.append(int(y[0]))
    torch.cuda.synchronize()
    elapsed=perf_counter()-tick
    if any(s.dead for s in (model,model.cf,model.cp)) or model.ft.over or calls!=(len(windows) if changed else 0):
        raise ValueError("timed archive failed or skipped a real CPU actor call")
    return dict(seconds=elapsed,restore_s=restored-tick,first_s=times[0],p95_s=float(np.percentile(times,95)),
                peak_reserved=torch.cuda.max_memory_reserved(),calls=calls,labels=labels,work=replay.activity(model))


def timing(stock,trial,cases,index):
    pin=json.loads(Path("smoke.json").read_text())
    if pin["source"]!=sha("source.sha256") or pin["archive"]!=Path("archive.sha256").read_text().split()[0] or pin["held_labels_opened"]:
        raise ValueError("prior genuine recognized and unrecognized archive smoke is required")
    name=ARM_ORDER[index]
    x=cases[name][0]
    check=pin["result"][name]
    timed_run(stock,x,False)
    timed_run(trial,x,True)
    order=("stock","hard","hard","stock") if index==0 else ("hard","stock","stock","hard")
    runs=[]
    for arm in order:
        result=timed_run(trial if arm=="hard" else stock,x,arm=="hard")
        if result["labels"]!=check[arm] or result["work"]!=check[arm+"_work"]:
            raise ValueError("warm official timing did not retain the verified decisions and real work")
        runs.append(result)
    a,b=(np.array([r["seconds"] for j,r in zip(order,runs,strict=True) if j==k]) for k in ("stock","hard"))
    ratio,worst=float(b.mean()/a.mean()),float((b/a).max())
    return dict(fixture=name,order=order,runs=runs,ratio=ratio,worst=worst,paired=(b/a).tolist(),
                meets_timing=ratio<=1.12 and worst<=1.15)


def main(args):
    if torch.__version__!="2.14.1+cu130" or "RTX 2080 Ti" not in torch.cuda.get_device_name():
        raise ValueError("pinned clean RTX image required")
    torch.set_num_threads(1)
    cases,channels,original=fixtures()
    with TemporaryDirectory() as root:
        a,b=load(args.stock,args.trial,Path(root),channels)
        result=smoke(a,b,cases) if args.mode=="smoke" else timing(a,b,cases,args.index)
    p=Path("smoke.json" if args.mode=="smoke" else f"timing_{args.index}.json")
    with p.open("x") as f:
        json.dump(dict(source=sha("source.sha256"),archive=sha(args.trial),stock=STOCK,actors=LOCK,
                       original_external_cold=original,result=result,held_labels_opened=False),
                  f,sort_keys=True,allow_nan=False)
        f.write("\n")
    p.chmod(0o444)
    print("PASS F-C25p",args.mode,args.index,sha(p),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("smoke","timing"))
    p.add_argument("index",type=int,choices=(0,1))
    p.add_argument("stock",type=Path)
    p.add_argument("trial",type=Path)
    main(p.parse_args())
