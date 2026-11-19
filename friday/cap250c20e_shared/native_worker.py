"""Use the real stock upstream once and seal both complete post-treatment decisions."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import replay
from frames import layout
from mirror import run_stream
from settings import ACTORS, sha, stamp
from worker import authentic_stock, stream


def opaque(pupil_path, mode, index, choices, who):
    assert not pupil_path.stat().st_mode & 0o222
    output={}
    with np.load(pupil_path,allow_pickle=False) as z:
        info=json.loads(str(z["metadata"]))
        assert info==dict(source=sha("source.sha256"),mode=mode,index=index,who=who,actors=list(ACTORS),
                          keys=list(choices),reset_prefix=True,held_labels_opened=False)
        for key,(eeg,rows) in choices.items():
            np.testing.assert_array_equal(z[key+"|rows"],rows)
            assert str(z[key+"|x"])==stamp(eeg[rows].numpy())
            p=z[key+"|p"]
            assert p.shape==(len(rows),2,3) and (p>=0).all() and np.isfinite(p).all()
            np.testing.assert_allclose(p.sum(2),1.,rtol=0,atol=1e-12)
            output[key]=p
    return output


def main(mode,index):
    assert torch.__version__=="2.14.1+cu130" and "RTX 2080 Ti" in torch.cuda.get_device_name()
    choices,((input_eeg,rows),archive,source),channels,who=layout(mode,index)
    path=Path(f"pupils_{mode}_{index}.npz")
    pupils=opaque(path,mode,index,choices,who)
    template=replay.load(Path("/package"),channels)
    assert template.routed==1 and template.r15 and not template.v6r and template.ft.cap==250
    base=stream(template,(),input_eeg,rows,"stock")
    authentic_stock(base,archive)
    if mode=="produce":
        witness=json.loads(Path("smoke.json").read_text())
        assert witness["passed"] and witness["source"]==sha("source.sha256")
        assert not witness["held_labels_opened"] and witness["mode"]=="smoke"
    rows_by={}
    for key,(eeg,ix) in choices.items():
        stock,hard,proof=run_stream(template,eeg,ix,pupils[key],zero=mode=="smoke")
        y0,y1=(b["terminal"].argmax(1).tolist() for b in (stock,hard))
        act=proof["activity"]
        if key=="P10-3":
            expected=json.loads(Path("/known.json").read_text())["streams"]["first"][key]
            assert y0==expected["labels"][:31] and proof["stop"]==10 and act["recognized"]==10
            assert act["ft_steps"]==0 and proof["warm_mix"]==0
        else:
            assert proof["stop"] is None and act["recognized"]==-1
            assert proof["warm_mix"]>0 and act["ft_steps"]>0 and proof["candidate_fuses"]>0
        if key.endswith("|0"):
            assert y0==archive["labels"] and act==archive["activity"]
        rows_by[key]=dict(rows=ix.tolist(),stock=y0,hard=y1, **proof)
        print("PASS F-C20s native",key,proof["max_mix"],flush=True)
    assert not template.v5hist and template.cp.t==0 and not template.dead
    device=torch.cuda.get_device_name()
    name=Path("smoke.json" if mode=="smoke" else f"person_{index}.json")
    info=dict(passed=True,mode=mode,person=index,who=who,source=sha("source.sha256"),
              pupil=sha(path),archive=sha(source),held_labels_opened=False,shared_protected=True,
              distinct_outer=True,device=device,sessions=rows_by,original_labels=base[0]["labels"])
    with name.open("x") as f:
        json.dump(info,f,sort_keys=True,allow_nan=False)
        f.write("\n")
    name.chmod(0o444)
    print("PASS F-C20s",mode,index,sha(name),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("smoke","produce"))
    p.add_argument("index",type=int,choices=range(9))
    a=p.parse_args()
    main(a.mode,a.index)
