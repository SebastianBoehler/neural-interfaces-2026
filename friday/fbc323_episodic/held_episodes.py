"""Verify source episodic work and independently replay the original static actor."""
import argparse
import json

import numpy as np
import torch

from anchors import DATA_SHA, ORIGIN_SHA, PEOPLE, RUNTIME, row_sha, sha, split
from episodes import ARMS, COUNT, ETA, HERE, KINDS, MODES, TOL, WEIGHT
from held import frozen_member
from held_distill import replay
from phase_source import F16_BANK, authenticated
from replay_f13 import ORDER_SHA, ORDERS, origin_report, raw_inputs


ACTOR = "d32d01f71b9eef27c383973238f02810486fd5b2988155e9d16039d3dfeb043a"
F16 = HERE.parent / "ni26_friday_s1c_distill_20261118"


def paired(held, seed, smoke, subject, visit):
    if sha(HERE/"actor.sha256")!=ACTOR or sha(F16/"before_labels.sha256")!=F16_BANK:
        raise ValueError("F30 actor or signed F16 blind bank differs")
    source, streams, people = split(subject,visit,held)
    base,_ = authenticated(F16,F16/"checkpoints",held,seed,smoke,source,streams,people)
    tag=f"{'smoke_' if smoke else ''}{seed}_{held}"
    fixed=dict(held=held,seed=seed,epochs=base["epochs"],scale=base["scale"],data=DATA_SHA,origin=ORIGIN_SHA,
               runtime=RUNTIME,actor=ACTOR,source_rows=row_sha(source),source_people=people,batches=base["batches"],
               initial=base["initial"],parameters=347903,weights_origin=base["checkpoint"],steps=base["steps"],
               held_labels_opened=False,source_labels_only=True,rng_isolated=True,step_size=ETA,q_gradients_nonzero=0,
               torch=base["torch"],braindecode=base["braindecode"],marginals_exact=True,roles_disjoint=True)
    result, active = {}, {}
    for kind in KINDS:
        path=HERE/"checkpoints"/f"{kind}_{tag}.pt"
        meta=json.loads(path.with_suffix(".json").read_text())
        expected=dict(fixed,kind=kind,checkpoint=sha(path),episode_weight=0. if kind=="hard" else WEIGHT)
        if (any(meta[k]!=v for k,v in expected.items()) or "H100" not in meta["device"] or
                not np.isfinite([meta[k] for k in ("first_loss","last_loss","source_error","max_gradient")]).all() or
                not meta["source_error"]>0 or not meta["max_gradient"]>0 or meta["state"]==meta["initial"] or
                not 6<=meta["episode_steps"]<=base["steps"] or meta["episode_steps"]/base["steps"]<.15 or
                meta["support_exposure"]!=meta["query_exposure"] or meta["support_exposure"]!=2*COUNT*meta["episode_steps"]):
            raise ValueError(f"original recipe or matched source episode controls differ for {kind}")
        evidence={k:meta[k] for k in ("episode_sha","episode_steps","support_exposure","query_exposure","diagnostic_steps")}
        if not active:
            active=evidence
        elif active!=evidence:
            raise ValueError("cross/within did not receive identical source support and query rows")
        if kind=="hard":
            if meta["state"]!=base["state"] or meta["checkpoint"]!=base["checkpoint"] or meta["actual_episodes"]:
                raise ValueError("exact hard source anchor differs")
        else:
            signals=[meta[k] for k in ("support_gradient","query_gradient","auxiliary_mean","temporary_move")]
            if (not np.isfinite(signals).all() or min(signals)<=0 or meta["actual_episodes"]!=meta["episode_steps"] or
                    not 0<=meta["analytic_error"]<=TOL):
                raise ValueError("matched extra relation gradients did not train the source features")
        result[kind]=path,meta
    if len({m["state"] for _,m in result.values()})!=3 or len({m["device"] for _,m in result.values()})!=1:
        raise ValueError("unpaired or duplicate source models")
    return result


def run(held, seed, smoke):
    torch.set_num_threads(4)
    origin_report()
    x, subject, person, stream, perms=raw_inputs()
    visit=np.asarray([v.rsplit("|",1)[-1] for v in stream])
    rows=np.flatnonzero(person==held)
    decisions,evidence={},{}
    for kind,(path,meta) in paired(held,seed,smoke,subject,visit).items():
        bank, proof=replay(*frozen_member(path,meta),x,stream,rows,perms,meta["scale"])
        for mode,value in zip(MODES,bank,strict=True):
            decisions[f"{kind}_{mode}"]=value
        evidence[kind]=dict(proof,checkpoint=meta["checkpoint"],state=meta["state"],episode_sha=meta["episode_sha"],
                            receipt=sha(path.with_suffix(".json")))
    if tuple(decisions)!=ARMS or not all(p["clips"] for p in evidence.values()):
        raise ValueError("matched causal held replay is incomplete")
    record=dict(held=held,seed=seed,actor=ACTOR,scorer=sha(HERE/"score.sha256"),data=DATA_SHA,
                origin=ORIGIN_SHA,orders=list(ORDERS),order_sha=ORDER_SHA,held_labels_opened=False,evidence=evidence)
    if not smoke:
        name=f"{held}_s{seed}.npz"
        with np.load(F16/"folds"/name,allow_pickle=False) as original:
            if not np.array_equal(original["index"],rows) or any(not np.array_equal(
                    original[f"hard_{m}"],decisions[f"hard_{m}"]) for m in MODES):
                raise ValueError("F30 hard held decisions differ from exact F16")
        path=HERE/"folds"/name
        with path.open("xb") as out:
            np.savez_compressed(out,index=rows,metadata=json.dumps(record,sort_keys=True),**decisions)
        with path.with_suffix(".sha256").open("x") as out:
            out.write(sha(path)+"\n")
    print(json.dumps(record,sort_keys=True),flush=True)
    print(f"PASS F30 held {held} seed {seed}",flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("held",choices=PEOPLE)
    cli.add_argument("seed",type=int,choices=(0,1))
    cli.add_argument("--smoke",action="store_true")
    args=cli.parse_args()
    run(args.held,args.seed,args.smoke)
