"""Qualify the original compact native F35e models in the CPU worker image."""
import argparse
import json

import numpy as np
import torch

from all_source import BUNDLE,F35,HERE,scope
from anchors import DATA_SHA,row_sha,sha
from export_checks import check,reference,reset
from external_checks import ACTOR,authenticated_group
from held import frozen_member
from incremental_model import export_actor
from recipe import GPU
from replay_f13 import DATA,origin_report
from source_scope import F16E_LOCK


def run(seed,smoke):
    if str(torch.__version__)!="2.14.1+cu130":
        raise ValueError("wrong original CPU worker image")
    torch.set_num_threads(1)
    origin_report()
    x,rows,streams,people,_=scope(DATA)
    names=np.unique(streams)
    picked=[np.flatnonzero(streams==n)[:48] for n in (names[0],names[-1])]
    sample=[np.ascontiguousarray(x[i]) for i in picked]
    if any(a.shape!=(48,41,480) for a in sample):
        raise ValueError("real source-only export replay is incomplete")
    out=HERE/("smoke_exports" if smoke else "exports")
    results={}
    for kind,(path,meta) in authenticated_group(seed,smoke,rows,streams,people).items():
        model,layer=frozen_member(path,meta)
        truth,active=reference(model,layer,sample,meta["scale"])
        if active<=0:
            raise ValueError("real source fixture did not activate bounded normalization")
        actor=export_actor(model,layer,meta["scale"],torch.from_numpy(sample[0][:1]))
        with torch.jit.optimized_execution(False):
            reset(actor)
        folder=out/kind
        folder.mkdir(parents=True,exist_ok=True)
        target=folder/f"raw_s1c_s{seed}.pt"
        pending=folder/f".{target.name}.pending"
        if target.exists() or pending.exists():
            raise FileExistsError(target)
        torch.jit.save(actor,str(pending))
        cpu=torch.jit.load(str(pending),map_location="cpu").eval()
        proof=check(cpu,sample,truth)
        pending.rename(target)
        result=dict(proof,path=str(target),sha256=sha(target),state=meta["state"],checkpoint=sha(path),
                    source_receipt=sha(path.with_suffix(".json")),bounded_active=active,tape=meta["tape"],trajectory=meta.get("trajectory"))
        results[kind]=result
    receipt=dict(seed=seed,smoke=smoke,device="cpu",fit_device=GPU,torch=str(torch.__version__),actor=ACTOR,
                 exporter=sha(HERE/"export.sha256"),f16e_lock=F16E_LOCK,f25e=BUNDLE,f35=F35,input=DATA_SHA,source_rows=row_sha(rows),
                 probe_rows=[row_sha(i) for i in picked],scherer_exposed=False,gold_opened=False,results=results)
    path=out/f"s{seed}.json"
    with path.open("x") as handle:
        handle.write(json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(f"PASS F35e CPU seed {seed} receipt {sha(path)}",flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("seed",type=int,choices=(0,1))
    cli.add_argument("--smoke",action="store_true")
    args=cli.parse_args()
    run(args.seed,args.smoke)
