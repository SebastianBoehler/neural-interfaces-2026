"""Authenticate the nested source experiment and independently replay native S1C."""
import argparse
import json

import numpy as np
import torch

from anchors import DATA_SHA, ORIGIN_SHA, PEOPLE, RUNTIME, row_sha, sha, split
from bank import TEACHER_SHA, receipt
from held import frozen_member
from held_distill import replay
from phase_source import F16_BANK, authenticated
from relational import ARMS, BETA, HERE, KINDS, MODES, WEIGHT, ids, tag_for
from replay_f13 import ORDER_SHA, ORDERS, origin_report, raw_inputs


ACTOR = "8ab663afcf72b5a8e6668e379f1e00fda27effe983beafba48cf84e2373298c3"
F16 = HERE.parent / "ni26_friday_s1c_distill_20261118"


def paired(held, seed, smoke, subject, visit):
    if sha(HERE/"actor.sha256")!=ACTOR or sha(HERE/"teacher.sha256")!=TEACHER_SHA or sha(F16/"before_labels.sha256")!=F16_BANK:
        raise ValueError("F28 source, teacher or authentic F16 blind bank differs")
    person = ids(subject)
    stream = np.char.add(np.char.add(subject,"|"),visit)
    source, source_stream, parents = split(subject,visit,held)
    base,_ = authenticated(F16,F16/"checkpoints",held,seed,smoke,source,source_stream,parents)
    query, teachers = [], {}
    for g in range(3):
        _, rows, proof, _ = receipt(HERE/"teachers",person,stream,held,seed,g,smoke,base)
        teachers[str(g)] = proof
        query.append(rows)
    if not np.array_equal(np.sort(np.concatenate(query)), source):
        raise ValueError("nested teacher query groups do not exhaust the nine independent source people")
    fixed = dict(held=held,seed=seed,epochs=base["epochs"],scale=base["scale"],data=DATA_SHA,origin=ORIGIN_SHA,
                 runtime=RUNTIME,actor=ACTOR,teacher_actor=TEACHER_SHA,teachers=teachers,source_rows=row_sha(source),
                 source_people=parents,batches=base["batches"],initial=base["initial"],parameters=347903,
                 weights_origin=base["checkpoint"],steps=base["steps"],held_labels_opened=False,source_labels_only=True,
                 rng_isolated=True,beta=BETA,q_gradients_nonzero=0,torch=base["torch"],braindecode=base["braindecode"],
                 marginal_exact=True)
    result, active = {}, {}
    for kind in KINDS:
        path=HERE/"checkpoints"/f"{kind}_{tag_for(held,seed,smoke)}.pt"
        meta=json.loads(path.with_suffix(".json").read_text())
        expected=dict(fixed,kind=kind,checkpoint=sha(path),relation_weight=0. if kind=="hard" else WEIGHT)
        if (any(meta[k]!=v for k,v in expected.items()) or "H100" not in meta["device"] or
                not np.isfinite([meta[k] for k in ("first_loss","last_loss","source_error","max_gradient")]).all() or
                not meta["source_error"]>0 or not meta["max_gradient"]>0 or meta["state"]==meta["initial"] or
                not 6<=meta["pair_steps"]<=base["steps"] or meta["pair_steps"]/base["steps"]<.05 or
                meta["pair_count"]<100 or not 1e-6<meta["minimum_target_move"]<=2):
            raise ValueError(f"original recipe or source pair controls differ for {kind}")
        evidence={k:meta[k] for k in ("pair_tape","pair_steps","pair_count","diagnostic_steps","minimum_target_move")}
        if not active:
            active=evidence
        elif active!=evidence:
            raise ValueError("literal/sham did not see identical matched pair supervision")
        if kind=="hard":
            if meta["state"]!=base["state"] or meta["checkpoint"]!=base["checkpoint"] or meta["actual_pairs"]:
                raise ValueError("exact hard source anchor differs")
        else:
            signals=[meta[k] for k in ("auxiliary_gradient","feature_gradient","auxiliary_mean")]
            if not np.isfinite(signals).all() or min(signals)<=0 or meta["actual_pairs"]!=meta["pair_steps"]:
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
        evidence[kind]=dict(proof,checkpoint=meta["checkpoint"],state=meta["state"],pair_tape=meta["pair_tape"],
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
                raise ValueError("F28 hard held decisions differ from exact F16")
        path=HERE/"folds"/name
        with path.open("xb") as out:
            np.savez_compressed(out,index=rows,metadata=json.dumps(record,sort_keys=True),**decisions)
        with path.with_suffix(".sha256").open("x") as out:
            out.write(sha(path)+"\n")
    print(json.dumps(record,sort_keys=True),flush=True)
    print(f"PASS F28 held {held} seed {seed}",flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("held",choices=PEOPLE)
    cli.add_argument("seed",type=int,choices=(0,1))
    cli.add_argument("--smoke",action="store_true")
    args=cli.parse_args()
    run(args.held,args.seed,args.smoke)
