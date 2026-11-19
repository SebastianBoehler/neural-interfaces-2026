"""Independently replay the original native actor and training-only F36 fits."""
import argparse
import json

import numpy as np
import torch

from anchors import DATA_SHA,ORIGIN_SHA,PEOPLE,sha
from crossfade import ARMS,HERE,MODES
from held import frozen_member
from held_distill import replay
from recipe import GPU
from replay_f13 import ORDER_SHA,ORDERS,origin_report,raw_inputs
from pair_checks import ACTOR,F33,paired


def run(held,seed,smoke):
    torch.set_num_threads(4)
    origin_report()
    x,subject,person,streams,perms=raw_inputs()
    visit=np.asarray([v.rsplit("|",1)[-1] for v in streams])
    rows=np.flatnonzero(person==held)
    decisions,evidence={},{}
    for kind,(path,meta) in paired(held,seed,smoke,subject,visit).items():
        winner,proof=replay(*frozen_member(path,meta),x,streams,rows,perms,meta["scale"])
        for mode,labels in zip(MODES,winner,strict=True):
            decisions[f"{kind}_{mode}"]=labels
        evidence[kind]=dict(proof,checkpoint=meta["checkpoint"],state=meta["state"],receipt=sha(path.with_suffix(".json")),tape=meta["tape"])
    if tuple(decisions)!=ARMS or not all(v["clips"]>0 for v in evidence.values()):
        raise ValueError("incomplete held bounded activity")
    stamp=dict(held=held,seed=seed,actor=ACTOR,scorer=sha(HERE/"score.sha256"),data=DATA_SHA,origin=ORIGIN_SHA,
               orders=list(ORDERS),order_sha=ORDER_SHA,device=GPU,held_labels_opened=False,evidence=evidence)
    if not smoke:
        name=f"{held}_s{seed}.npz"
        with np.load(F33/"folds"/name,allow_pickle=False) as prior:
            if not np.array_equal(prior["index"],rows) or any(not np.array_equal(
                    decisions[f"hard_{m}"],prior[f"hard_{m}"]) for m in MODES):
                raise ValueError("real held hard A100 anchor differs")
        output=HERE/"folds"/name
        with output.open("xb") as out:
            np.savez_compressed(out,index=rows,metadata=json.dumps(stamp,sort_keys=True,allow_nan=False),**decisions)
        with output.with_suffix(".sha256").open("x") as out:
            out.write(sha(output)+"\n")
    print(json.dumps(stamp,sort_keys=True,allow_nan=False),flush=True)
    print(f"PASS F36 held {held} seed {seed}",flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("held",choices=PEOPLE)
    cli.add_argument("seed",type=int,choices=(0,1))
    cli.add_argument("--smoke",action="store_true")
    args=cli.parse_args()
    run(args.held,args.seed,args.smoke)
