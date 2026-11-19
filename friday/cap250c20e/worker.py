"""Profile and run the native cap250 plus its outer-only hard waveform variant."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

import replay
from adapter import Adapter
from benchmark_utils.streaming import fresh_model
from settings import ACTORS, ARMS, SEEDS, UPPER, schedule, sha, stamp


def close(a,b):
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    error=float(np.abs(a-b).max())
    assert error <= UPPER, error
    np.testing.assert_array_equal(a.argmax(-1), b.argmax(-1))
    return error


def stream(template, actors, eeg, rows, arm):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start=perf_counter()
    model=fresh_model(template)
    assert not model.v5hist and model.cp.t==0 and not model.dead
    tape=Adapter(model,arm,actors)
    original=model.cp.step
    scores=[]

    def observe(*args):
        out=original(*args)
        scores.append(out.copy())
        return out

    model.cp.step=observe
    created=perf_counter()
    labels,times=[],[]
    for row in rows:
        before=perf_counter()
        out=model.predict(eeg[int(row):int(row)+1].to("cuda:0"))
        assert out.shape==(1,) and out.dtype==torch.int64 and out.device.type=="cpu"
        assert not any(p.dead for p in (model,model.cf,model.cp))
        tape.record()
        labels.append(int(out[0]))
        times.append(perf_counter()-before)
    torch.cuda.synchronize()
    end=perf_counter()
    activity=replay.activity(model)
    assert set(labels)<={0,1,2} and activity["windows"]==len(rows) and activity["errors"]==activity["ft_over"]==0
    assert activity["ft_steps"]==sum(r["steps"] for r in tape.refits)==20*activity["ft_refits"]
    assert model.cp.stopped==(None if activity["recognized"] == -1 else activity["recognized"])
    assert len(scores)==len(rows)
    np.testing.assert_array_equal(np.asarray(scores).argmax(1),labels)
    audit=tape.summary()
    if arm=="hard":
        assert audit["raw_calls"]==len(rows) and len(tape.pupil)==len(rows) and audit["max_move"]>.01
        assert audit["warm_rows"]==(activity["ft_refits"]>0)*int(model.log6["ft"])
    return dict(labels=labels,activity=activity,seconds=end-start,restore_s=created-start,
                first_s=times[0],p95_s=float(np.percentile(times,95)),max_s=max(times),
                peak_reserved=torch.cuda.max_memory_reserved(),adapter=audit), dict(
                upstream=np.asarray(tape.trace),loss=tape.refits,terminal=np.asarray(scores),pupils=np.asarray(tape.pupil))


def same_upstream(r,q):
    stock,s=r
    test,t=q
    protection=("windows","recognized","ft_refits","ft_steps","ft_relabels","ft_capped","ft_over","errors")
    assert all(stock["activity"][k]==test["activity"][k] for k in protection)
    delta=close(s["upstream"],t["upstream"])
    assert len(s["loss"])==len(t["loss"])
    largest=0.
    for a,b in zip(s["loss"],t["loss"],strict=True):
        for k in ("age","raw","labels","steps"):
            assert a[k]==b[k],k
        change=float(np.max(np.abs(np.asarray(a["weights"])-b["weights"])))
        largest=max(largest,change)
    assert largest<=UPPER
    return dict(max_upstream=delta,max_loss_weight=largest,loss_rows=sum(r["age"] for r in s["loss"]))


def authentic_stock(r,expect):
    v=r[0]
    assert v["labels"]==expect["labels"] and v["activity"]==expect["activity"]


@torch.no_grad()
def cpu_pupils(eeg, rows, n):
    payload=[]
    for j in (0,1):
        actor=torch.jit.load(f"/actors/raw_s1c_s{j}.pt",map_location="cpu").eval()
        with torch.jit.optimized_execution(False):
            assert actor.reset().item()==0
            payload.append(np.stack([actor(eeg[int(r):int(r)+1].contiguous()).numpy()[0] for r in rows[:n]]))
    return np.stack(payload,axis=1)


def validate(template,actors,eeg,rows,other,anchor,channels):
    stock=stream(template,actors,eeg,rows,"stock")
    authentic_stock(stock,anchor)
    identity=stream(template,actors,eeg,rows,"identity")
    authentic_stock(identity,anchor)
    control=same_upstream(stock,identity)
    control["terminal_error"]=close(stock[1]["terminal"],identity[1]["terminal"])
    assert identity[0]["adapter"]["raw_calls"]==len(rows) and identity[0]["adapter"]["max_move"]==0
    hard=stream(template,actors,eeg,rows,"hard")
    match=same_upstream(stock,hard)
    assert match["loss_rows"]>0 and hard[0]["activity"]["recognized"]==-1 and hard[0]["activity"]["cp_fused"]>0
    assert hard[0]["adapter"]["warm_rows"]>0
    stream(template,actors,eeg,other[:17],"hard")
    prefix=stream(template,actors,eeg,rows[:31],"hard")
    np.testing.assert_array_equal(hard[1]["pupils"][:31],prefix[1]["pupils"])
    assert hard[0]["labels"][:31]==prefix[0]["labels"]
    prefix_error=close(hard[1]["terminal"][:31],prefix[1]["terminal"])
    cpu_error=close(cpu_pupils(eeg,rows,31),prefix[1]["pupils"])
    x,public,public_channels=replay.read_source()
    assert public_channels==channels
    known=dict(public)["P10-3"][:31]
    recorded=json.loads(Path("/known.json").read_text())["streams"]["first"]["P10-3"]
    assert known.tolist()==recorded["rows"][:31]
    original=stream(template,actors,x,known,"stock")
    proposed=stream(template,actors,x,known,"hard")
    assert original[0]["labels"]==recorded["labels"][:31] and original[0]["activity"]["recognized"]==10
    recognized=same_upstream(original,proposed)
    return dict(passed=True,rows=rows.tolist(),identity=identity[0],stock=stock[0],hard=hard[0],protection=match,
                control=control,recognized=dict(stock=original[0],hard=proposed[0],proof=recognized,rows=known.tolist()),
                prefix_terminal_error=prefix_error,cpu_pupil_error=cpu_error)


def all_rows(index, template, actors, eeg, who, plans, anchor):
    order0=plans[0][who+"|1"]
    assert order0.tolist()==anchor["rows"]
    cold=stream(template,actors,eeg,order0,"stock")
    authentic_stock(cold,anchor)
    warm={k:stream(template,actors,eeg,order0[:17],k)[0] for k in ARMS}
    assert warm["stock"]["labels"]==anchor["labels"][:17]
    values={}
    for j,s in enumerate(SEEDS):
        for i,day in enumerate((1,2)):
            key=f"{who}|{day}"
            rows=plans[s][key]
            seq=ARMS if (index+j+i)%2==0 else ARMS[::-1]
            bank={k:stream(template,actors,eeg,rows,k) for k in seq}
            proof=same_upstream(bank["stock"],bank["hard"])
            hard=bank["hard"][0]
            assert proof["loss_rows"]>0 and hard["activity"]["recognized"]==-1
            assert hard["adapter"]["warm_rows"]>0 and hard["activity"]["cp_fused"]>0
            values[f"{key}|{s}"]=dict(order=list(seq),rows=rows.tolist(),protection=proof,
                                    runs={k:v[0] for k,v in bank.items()})
            print("SEALED",key,s,seq,values[f"{key}|{s}"]["protection"],flush=True)
    return dict(cold=cold[0],cold_rows=order0.tolist(),warm=warm,sessions=values)


def main(mode,index):
    assert torch.__version__=="2.14.1+cu130" and "RTX 2080 Ti" in torch.cuda.get_device_name()
    assert "neural-interfaces-2026/.venv/" not in Path("/proc/self/maps").read_text()
    assert tuple(sha(f"/actors/raw_s1c_s{k}.pt") for k in (0,1))==ACTORS
    with np.load("/external.npz",allow_pickle=False) as z:
        eeg=torch.from_numpy(z["X"])
        people,visits,channels=z["subject"],z["session"],z["ch_names"].tolist()
    assert eeg.shape==(2130,41,480) and eeg.dtype==torch.float32 and torch.isfinite(eeg).all()
    who=sorted(set(people))[index]
    plans=schedule(people,visits)
    archive=Path("/rtx_pair" if index in (0,5) else "/rtx_other")/f"rtx_cap_{index}.json"
    old=json.loads(archive.read_text())
    assert old["person"]==index and old["arm"]=="cap" and "RTX 2080 Ti" in old["device"]
    anchor=old["sessions"][f"{who}|1|0"]
    assert anchor["rows"]==plans[0][who+"|1"].tolist()
    template=replay.load(Path("/package"),channels)
    assert template.perm is None and template.routed==1 and template.ft.on and template.ft.cap==250
    actors=tuple(torch.jit.load(f"/actors/raw_s1c_s{k}.pt",map_location="cuda:0").eval() for k in (0,1))
    source=sha("source.sha256")
    if mode=="smoke":
        assert who=="Scherer2015/G" and len(plans[0][who+"|1"])==120
        result=validate(template,actors,eeg,plans[0][who+"|1"],plans[0][who+"|2"],anchor,channels)
        suffix="smoke"
        smoke_sha=None
    else:
        witness=json.loads(Path("smoke.json").read_text())
        assert witness["passed"] and witness["source"]==source and not witness["held_labels_opened"]
        smoke_sha=sha("smoke.json")
        result=all_rows(index,template,actors,eeg,who,plans,anchor)
        suffix=f"person_{index}"
    assert not template.v5hist and template.cp.t==0 and not template.dead
    assert "neural-interfaces-2026/.venv/" not in Path("/proc/self/maps").read_text()
    path=Path(suffix+".json")
    with path.open("x") as f:
        json.dump(dict(**result,person=index,who=who,source=source,smoke=smoke_sha,archive=sha(archive),
                       held_labels_opened=False,torch=torch.__version__,gpu=torch.cuda.get_device_name()),f,
                  sort_keys=True,allow_nan=False)
        f.write("\n")
    path.chmod(0o444)
    print("PASS F-C20e",mode,sha(path),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("smoke","produce"))
    p.add_argument("index",type=int,choices=range(9))
    a=p.parse_args()
    main(a.mode,a.index)
