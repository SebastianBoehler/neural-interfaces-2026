"""Confirm F34 source fits and replay independently on untouched held waveforms."""
import argparse
import json

import numpy as np
import torch

from anchors import DATA_SHA,ORIGIN_SHA,PEOPLE,row_sha,sha,split
from envelope import ARMS,BASE,EXTRA,HERE,KINDS,MAX_RATIO,MAX_RESERVED,MODES,attach,parameter_hash,trainable
from group import ACTOR as F33A
from group import paired as anchor
from held_distill import replay
from recipe import BD,CPU_CUDA,GPU,TORCH,canary,historical,tag
from replay_f13 import ORDER_SHA,ORDERS,origin_report,raw_inputs
from source_targets import normalizer


ACTOR="56ca60c2d190b3b51005f09cd53d66243af92ba2a319e58544594d0e9a94e60c"
F33=HERE.parent/"ni26_friday_s1c_covariance_a100_whole_20261119"
F33_BLIND="f996e3a55d376f6856bbd07425632033924a175636acc21b332d1a4d3f4f0525"


def paired(held,seed,smoke,subject,visit):
    if sha(HERE/"actor.sha256")!=ACTOR or sha(F33/"before_labels.sha256")!=F33_BLIND:
        raise ValueError("F34 source or original A100 decision seal differs")
    rows,streams,people=split(subject,visit,held)
    recipe,batches,_=historical(held,seed,smoke,rows,streams,people)
    ref=anchor(held,seed,smoke,subject,visit)["hard"][1]
    env=canary()
    fixed=dict(held=held,seed=seed,epochs=recipe["epochs"],steps=recipe["steps"],scale=ref["scale"],initial=recipe["initial"],
               data=DATA_SHA,origin=ORIGIN_SHA,source_rows=row_sha(rows),source_people=people,batches=recipe["batches"],
               held_labels_opened=False,actor=ACTOR,f33a=F33A,weights_origin=ref["checkpoint"],
               device=GPU,torch=TORCH,braindecode=BD,rng_isolated=True,zero_identity=True,q_gradients_nonzero=0,
               diagnostic_rows=row_sha(rows[batches[0][0][:32]]),**{k:env[k] for k in CPU_CUDA})
    result={}
    for kind in KINDS:
        path=HERE/"checkpoints"/f"{kind}_{tag(held,seed,smoke)}.pt"
        meta=json.loads(path.with_suffix(".json").read_text())
        required=dict(fixed,kind=kind,checkpoint=sha(path),parameters=BASE+(0 if kind=="hard" else EXTRA))
        if (any(meta[k]!=v for k,v in required.items()) or not 0<meta["peak_reserved"]<=MAX_RESERVED or
                not np.isfinite([meta[k] for k in ("first_loss","last_loss","max_gradient")]).all() or
                not meta["max_gradient"]>0 or meta["state"]==meta["initial"]):
            raise ValueError("source receipt, fit exclusion or A100 native geometry differs")
        if kind=="hard":
            if any(meta[k]!=ref[k] for k in ("state","checkpoint")):
                raise ValueError("hard A100 anchor differs byte-for-byte")
        else:
            signals=[meta[k] for k in ("spatial_gradient","first_gradient","second_gradient","readout_gradient")]
            if (not np.isfinite(signals).all() or min(signals)<=1e-8 or
                    not 0<=meta["energy_error"]<=2e-6 or not 0<=meta["analytic_phase_error"]<=2e-6 or
                    not meta["real_phase_delta"]>=1e-3 or not meta["logit_swap"]>1e-4 or
                    meta["initial_branch"]==meta["final_branch"]):
                raise ValueError("fitted raw envelope or phase invariance evidence differs")
        result[kind]=path,meta
    members=[m for _,m in result.values()]
    if (len({m["state"] for m in members})!=3 or
            result["analytic"][1]["initial_branch"]!=result["real_power"][1]["initial_branch"]):
        raise ValueError("matched envelope branch identity differs")
    if smoke:
        ms=result["hard"][1]["warm_ms"]
        if any(m["hard_ms"]!=ms or not 0<m["warm_ms"]<=MAX_RATIO*ms for m in members):
            raise ValueError("A100 paired forward time differs")
    return result


def frozen(path,meta):
    import model322
    from encoder312 import fingerprint

    model322.SEED=meta["seed"]
    trained=torch.load(path,map_location="cpu",weights_only=True)
    if any(trained[k]!=meta[k] for k in ("kind","held","seed","scale")):
        raise ValueError("bad held payload identity")
    model=model322.Factor("session",False,"cpu")
    branch=attach(model,meta["kind"],meta["seed"])
    if branch is not None and parameter_hash(branch)!=meta["initial_branch"]:
        raise ValueError("independent raw branch initialization differs")
    model.load_state_dict(trained["state"],strict=True)
    model.eval()
    if (fingerprint(model)!=meta["state"] or trainable(model)!=meta["parameters"] or
            branch is not None and parameter_hash(branch)!=meta["final_branch"]):
        raise ValueError("frozen A100 envelope state differs")
    return model,normalizer(model)


def run(held,seed,smoke):
    torch.set_num_threads(4)
    origin_report()
    x,subject,person,streams,perms=raw_inputs()
    visit=np.asarray([v.rsplit("|",1)[-1] for v in streams])
    rows=np.flatnonzero(person==held)
    decisions,evidence={},{}
    for kind,(path,meta) in paired(held,seed,smoke,subject,visit).items():
        winner,proof=replay(*frozen(path,meta),x,streams,rows,perms,meta["scale"])
        for mode,labels in zip(MODES,winner,strict=True):
            decisions[f"{kind}_{mode}"]=labels
        evidence[kind]=dict(proof,checkpoint=meta["checkpoint"],state=meta["state"],receipt=sha(path.with_suffix(".json")))
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
    print(f"PASS F34 held {held} seed {seed}",flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("held",choices=PEOPLE)
    cli.add_argument("seed",type=int,choices=(0,1))
    cli.add_argument("--smoke",action="store_true")
    args=cli.parse_args()
    run(args.held,args.seed,args.smoke)
