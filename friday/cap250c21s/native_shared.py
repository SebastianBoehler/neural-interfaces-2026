"""Seal fixed-input predictions and separately diagnose the failed stock pair."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import replay
from matched import paired
from native21 import stream
from scope import SEEDS, external, known
from settings import sha
from worker import authentic_stock


def diagnosis(template,eeg,p0,p1,anchor):
    cold=stream(template,eeg,p0,"stock")
    authentic_stock(cold,anchor)
    for _ in range(2):
        stream(template,eeg,p0[:17],"stock")
    stream(template,eeg,p1[0],"stock")
    protected=("windows","recognized","ft_refits","ft_steps","ft_relabels","ft_capped","ft_over","errors")
    runs=[stream(template,eeg,p1[1],"stock") for _ in range(3)]
    first=runs[0][0]["activity"]
    assert all(all(x[0]["activity"][k]==first[k] for k in protected) for x in runs)
    drift=[]
    for i,j in ((0,1),(0,2),(1,2)):
        u,v=runs[i][1],runs[j][1]
        assert len(u["loss"])==len(v["loss"])
        for a,b in zip(u["loss"],v["loss"],strict=True):
            assert all(a[k]==b[k] for k in ("age","raw","labels","steps")),(i,j,a["age"])
        delta=np.abs(u["upstream"]-v["upstream"]).max(axis=(0,2))
        drift.append(dict(pair=[i,j],component_max=delta.tolist(),sinc_max=float(delta[5]),
                          prior_weight_max=max(float(np.abs(np.array(a["weights"])-b["weights"]).max())
                              for a,b in zip(u["loss"],v["loss"],strict=True)),
                          final_differences=int(np.count_nonzero(np.array(runs[i][0]["labels"])!=runs[j][0]["labels"]))))
    return dict(activity=first,repeat=drift,rows=p1[1].tolist(),reproduced=any(d["sinc_max"]>5e-5 for d in drift))


def main(mode,index):
    assert torch.__version__=="2.14.1+cu130" and "RTX 2080 Ti" in torch.cuda.get_device_name()
    assert "neural-interfaces-2026/.venv/" not in Path("/proc/self/maps").read_text()
    i=5 if mode=="smoke" else 3 if mode=="diagnose" else index
    eeg,who,channels,orders,anchor,archive=external(i)
    template=replay.load(Path("/package"),channels)
    assert template.routed==1 and template.r15 and not template.v6r and template.ft.cap==250
    p0=orders[0][who+"|1"]
    if mode=="diagnose":
        payload=diagnosis(template,eeg,p0,[orders[10411][f"{who}|{v}"] for v in (1,2)],anchor)
    else:
        cold=stream(template,eeg,p0,"stock")
        authentic_stock(cold,anchor)
        payload=dict(cold=cold[0],cold_rows=p0.tolist(),results={})
        if mode=="smoke":
            x,r,expected=known(channels)
            known_row=paired(template,x,r,prefix=True,expected_stock=expected)
            payload["recognized"]=known_row
            assert known_row["route"]["routed"]==21 and known_row["protected"]["stop"]==10
            g=paired(template,eeg,p0,prefix=True,expected_stock=anchor["labels"])
            payload["unrecognized"]=g
            assert g["hook"]["warm_rows"]==100 and g["hook"]["fbl_warm_delta"]>.01
            assert g["trial_fuses"]>0 and not g["route"]["routed"]
        else:
            smoke=json.loads(Path("smoke.json").read_text())
            assert smoke["passed"] and smoke["source"]==sha("source.sha256") and not smoke["held_labels_opened"]
            for j in SEEDS:
                for v in (1,2):
                    key=f"{who}|{v}"
                    q=paired(template,eeg,orders[j][key])
                    assert q["hook"]["warm_rows"]>0 and q["trial_fuses"]>0 and not q["route"]["routed"]
                    assert q["protected"]["activity"]["recognized"]==-1
                    payload["results"][f"{key}|{j}"]=q
                    print("SEALED F-C21s",key,j,q["terminal_change"],flush=True)
    assert not template.v5hist and not template.cp.t and not template.dead
    path=Path(mode+".json" if mode!="produce" else f"person_{i}.json")
    with path.open("x") as f:
        json.dump(dict(passed=True,source=sha("source.sha256"),index=i,person=who,archive=archive,scope=mode,
                  device=torch.cuda.get_device_name(),held_labels_opened=False,**payload),f,sort_keys=True,allow_nan=False)
        f.write("\n")
    path.chmod(0o444)
    print("PASS F-C21",mode,i,sha(path),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("smoke","produce","diagnose"))
    p.add_argument("index",type=int,choices=range(9))
    a=p.parse_args()
    main(a.mode,a.index)
