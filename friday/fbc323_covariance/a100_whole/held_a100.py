"""Replay the independent A100 fits and verify exact bare-versus-hard decisions."""
import argparse
import json

import numpy as np
import torch

from anchors import DATA_SHA,ORIGIN_SHA,PEOPLE,sha
from covariance import ARMS,MODES
from group import ACTOR,paired
from held_covariance import frozen_member
from held_distill import replay
from recipe import GPU,HERE
from replay_f13 import ORDER_SHA,ORDERS,origin_report,raw_inputs


BARE_ARMS=tuple(f"bare_{m}" for m in MODES)
ALL_ARMS=ARMS+BARE_ARMS


def run(held,seed,smoke):
    torch.set_num_threads(4)
    origin_report()
    x,subject,person,stream,perms=raw_inputs()
    visit=np.asarray([v.rsplit("|",1)[-1] for v in stream])
    rows=np.flatnonzero(person==held)
    decisions,evidence={},{}
    for kind,(path,meta) in paired(held,seed,smoke,subject,visit).items():
        labels,proof=replay(*frozen_member(path,meta),x,stream,rows,perms,meta["scale"])
        for mode,result in zip(MODES,labels,strict=True):
            decisions[f"{kind}_{mode}"]=result
        evidence[kind]=dict(proof,checkpoint=meta["checkpoint"],state=meta["state"],permutation=meta.get("permutation"),
                            receipt=sha(path.with_suffix(".json")))
    if tuple(decisions)!=ALL_ARMS or not all(p["clips"]>0 for p in evidence.values()):
        raise ValueError("missing bounded replay or fitted control")
    if any(not np.array_equal(decisions[f"hard_{mode}"],decisions[f"bare_{mode}"]) for mode in MODES):
        raise ValueError("independent A100 hard reference predictions differ")
    meta=dict(held=held,seed=seed,actor=ACTOR,scorer=sha(HERE/"score.sha256"),data=DATA_SHA,origin=ORIGIN_SHA,
              orders=list(ORDERS),order_sha=ORDER_SHA,device=GPU,exact_bare=True,held_labels_opened=False,evidence=evidence)
    if not smoke:
        path=HERE/"folds"/f"{held}_s{seed}.npz"
        with path.open("xb") as out:
            np.savez_compressed(out,index=rows,metadata=json.dumps(meta,sort_keys=True,allow_nan=False),**decisions)
        with path.with_suffix(".sha256").open("x") as out:
            out.write(sha(path)+"\n")
    print(json.dumps(meta,sort_keys=True,allow_nan=False),flush=True)
    print(f"PASS F33a held {held} seed {seed}",flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("held",choices=PEOPLE)
    cli.add_argument("seed",type=int,choices=(0,1))
    cli.add_argument("--smoke",action="store_true")
    args=cli.parse_args()
    run(args.held,args.seed,args.smoke)
