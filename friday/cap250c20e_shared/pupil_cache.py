"""Run both live TorchScript pupils on RTX before loading the separate stock process."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from frames import layout
from settings import ACTORS, sha, stamp


@torch.no_grad()
def session(templates, eeg, rows):
    actors=[copy.deepcopy(a) for a in templates]
    p=[]
    with torch.jit.optimized_execution(False):
        assert all(a.reset().item()==0 for a in actors)
        for r in rows:
            x=eeg[int(r):int(r)+1].to("cuda:0").contiguous()
            pair=[a(x) for a in actors]
            assert all(q.shape==(1,3) and q.dtype==torch.float64 for q in pair)
            p.append(np.stack([q.cpu().numpy()[0] for q in pair]))
    out=np.asarray(p)
    assert out.shape==(len(rows),2,3) and np.isfinite(out).all() and (out>=0).all()
    np.testing.assert_allclose(out.sum(2),1.,rtol=0,atol=1e-12)
    return out


def run(mode,index):
    assert torch.__version__=="2.14.1+cu130" and "RTX 2080 Ti" in torch.cuda.get_device_name()
    assert tuple(sha(f"/actors/raw_s1c_s{i}.pt") for i in (0,1))==ACTORS
    actors=tuple(torch.jit.load(f"/actors/raw_s1c_s{i}.pt",map_location="cuda:0").eval() for i in (0,1))
    choices,_,_,who=layout(mode,index)
    result={}
    keys=list(choices)
    for key,(eeg,rows) in choices.items():
        p=session(actors,eeg,rows)
        stranger=keys[(keys.index(key)+1)%len(keys)]
        x,other=choices[stranger]
        session(actors,x,other[:17])
        np.testing.assert_array_equal(p[:17],session(actors,eeg,rows[:17]))
        result[key+"|p"]=p
        result[key+"|rows"]=rows
        result[key+"|x"]=np.array(stamp(eeg[rows].numpy()))
    meta=dict(source=sha("source.sha256"),mode=mode,index=index,who=who,actors=ACTORS,keys=keys,
              reset_prefix=True,held_labels_opened=False)
    path=Path(f"pupils_{mode}_{index}.npz")
    with path.open("xb") as f:
        np.savez_compressed(f,metadata=json.dumps(meta,sort_keys=True),**result)
    path.chmod(0o444)
    print("PASS F-C20s independent RTX pupils",mode,index,sha(path),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("smoke","produce"))
    p.add_argument("index",type=int,choices=range(9))
    a=p.parse_args()
    run(a.mode,a.index)
