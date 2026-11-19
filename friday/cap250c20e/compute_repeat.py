"""Separate effects of extra CUDA model execution from the actual outer substitution."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

import replay
import worker
from adapter import Adapter
from settings import ACTORS, UPPER, schedule, sha, stamp

FAMILIES=("mix","shadow","v5","label","fb","fbl","fbraw","sinc")


def runtime():
    return dict(optimize=torch._C._get_graph_executor_optimize(), device=torch.cuda.current_device(),
                stream=torch.cuda.current_stream().cuda_stream, default_stream=torch.cuda.default_stream().cuda_stream,
                deterministic=torch.are_deterministic_algorithms_enabled(), cudnn=torch.backends.cudnn.deterministic,
                benchmark=torch.backends.cudnn.benchmark, cudnn_tf32=torch.backends.cudnn.allow_tf32,
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32, plans=int(torch.backends.cuda.cufft_plan_cache.size),
                allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved())


def slots(actors):
    return {v.untyped_storage().data_ptr() for actor in actors for v in
            list(actor.parameters())+list(actor.buffers()) if v.numel()}


class Audited(Adapter):
    def __init__(self, model, arm, templates):
        super().__init__(model,arm,templates)
        self.compute_only=arm=="identity"
        if self.compute_only:
            self.actors=tuple(deepcopy(a) for a in templates)
            with torch.no_grad(),torch.jit.optimized_execution(False):
                assert all(a.reset().item()==0 for a in self.actors)
        self.runtime=[]
        self.alias_checks=0

    def pupil_q(self, raw):
        monitor=len(self.pupil) in (0,19,20,21,31,63,119)
        start=runtime() if monitor else None
        out=super().pupil_q(raw)
        if monitor:
            end=runtime()
            semantic=set(start)-{"plans","allocated","reserved"}
            assert all(start[k]==end[k] for k in semantic)
            fit=self.model.ft.nets
            if fit is not None:
                assert not slots(fit.values()) & slots(self.actors)
                self.alias_checks+=1
            assert not slots(self.actors[:1]) & slots(self.actors[1:])
            self.runtime.append(dict(age=len(self.pupil)-1,before=start,after=end))
        return out

    def mix6(self, mix, members, warm):
        if not members:
            return super().mix6(mix,members,warm)
        refs=(mix.copy(),{i:stamp(v) for i,v in members.items()}, {k:stamp(v) for k,v in warm.items()})
        actors=self.actors
        if self.compute_only:
            self.pupil_q(self.window)
            self.actors=()
        try:
            out=super().mix6(mix,members,warm)
        finally:
            self.actors=actors
        np.testing.assert_array_equal(refs[0],mix)
        assert refs[1]=={i:stamp(v) for i,v in members.items()}
        assert refs[2]=={k:stamp(v) for k,v in warm.items()}
        return out

    def summary(self):
        return dict(super().summary(),runtime=self.runtime,alias_checks=self.alias_checks,pupil_calls=len(self.pupil))


def classify(a,b):
    ra,rb=a[1],b[1]
    deviation=np.abs(ra["upstream"]-rb["upstream"]).max(axis=2)
    detail={}
    for k,v in zip(FAMILIES,deviation.T,strict=True):
        changed=np.flatnonzero(v>1e-9)
        above=np.flatnonzero(v>UPPER)
        recent=[e["age"] for e in ra["loss"]]
        detail[k]=dict(max=float(v.max()),first=int(changed[0]) if changed.size else None,
                       first_above=int(above[0]) if above.size else None,
                       last_refit_before_max=max((n for n in recent if n<=int(v.argmax())),default=None),
                       max_age=int(v.argmax()) if v.max()>0 else None)
    return dict(statistics=detail,loss_exact=ra["loss"]==rb["loss"],
                older_bitwise=np.array_equal(ra["upstream"][:,:4],rb["upstream"][:,:4]),
                activity={k:[a[0]["activity"][k],b[0]["activity"][k]] for k in a[0]["activity"]},
                terminal_decisions_changed=int((ra["terminal"].argmax(1)!=rb["terminal"].argmax(1)).sum()))


def main():
    assert "RTX 2080 Ti" in torch.cuda.get_device_name() and torch.__version__=="2.14.1+cu130"
    assert tuple(sha(f"/actors/raw_s1c_s{k}.pt") for k in (0,1))==ACTORS
    with np.load("/external.npz",allow_pickle=False) as z:
        eeg=torch.from_numpy(z["X"])
        subject,days,channels=z["subject"],z["session"],z["ch_names"]
    who="Scherer2015/E|1"
    plan=schedule(subject,days)
    rows=plan[9411][who]
    old=json.loads(Path("/rtx_other/rtx_cap_3.json").read_text())["sessions"][who+"|0"]
    assert len(rows)==120 and old["rows"]==plan[0][who].tolist()
    model=replay.load(Path("/package"),channels.tolist())
    actors=tuple(torch.jit.load(f"/actors/raw_s1c_s{k}.pt",map_location="cuda:0").eval() for k in (0,1))
    cold=worker.stream(model,actors,eeg,plan[0][who],"stock")
    worker.authentic_stock(cold,old)
    names=("stock","identity","hard","hard","identity","stock")
    worker.Adapter=Audited
    runs=[]
    try:
        narrow=worker.stream(model,actors,eeg,rows[:31],"identity")
        assert narrow[0]["adapter"]["pupil_calls"]==31 and narrow[0]["adapter"]["max_move"]==0
        assert narrow[0]["adapter"]["alias_checks"]>0 and len(narrow[1]["loss"])>0
        print("PASS real 31-window compute-only preflight",flush=True)
        for name in names:
            runs.append(worker.stream(model,actors,eeg,rows,name))
            print("E compute-only witness",name,runs[-1][0]["adapter"]["pupil_calls"],flush=True)
    finally:
        worker.Adapter=Adapter
    for name,(r,p) in zip(names,runs,strict=True):
        expected=0 if name=="stock" else len(rows)
        assert r["adapter"]["pupil_calls"]==expected
        if name=="identity":
            assert r["adapter"]["max_move"]==0 and r["adapter"]["alias_checks"]>0
        if name=="hard":
            assert r["adapter"]["alias_checks"]>0 and r["adapter"]["max_move"]>.01
    pair=[(0,5)]+[(s,c) for s in (0,5) for c in (1,2,3,4)]
    results={f"{a}|{b}":classify(runs[a],runs[b]) for a,b in pair}
    payload=dict(rows=rows.tolist(),names=list(names),runs=[r[0] for r in runs],pairs=results,
                 held_labels_opened=False,source=sha("source.sha256"),stock_archive=sha("/rtx_other/rtx_cap_3.json"))
    output=Path("compute_repeat.json")
    with output.open("x") as f:
        json.dump(payload,f,sort_keys=True,allow_nan=False)
        f.write("\n")
    output.chmod(0o444)
    print("PASS F-C20e compute-only diagnostic",sha(output),flush=True)


if __name__=="__main__":
    main()
