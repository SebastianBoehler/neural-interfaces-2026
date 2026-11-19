"""Authenticate gain-invariant source pretext and independently replay native hard inference."""
import argparse
import json
from math import log

import numpy as np
import torch

from anchors import DATA_SHA, ORIGIN_SHA, PEOPLE, RUNTIME, row_sha, sha, split
from order_pretext import ARMS,HERE,KINDS,MODES,PARAMETERS,TOL,WEIGHT,tape
from held import frozen_member
from held_distill import replay
from phase_source import F16_BANK, authenticated
from replay_f13 import ORDER_SHA, ORDERS, origin_report, raw_inputs


ACTOR = "832bf16a1af007cadef7742a2ee3bb08cd27c51a9dccf4ebd9a691f6be38e24d"
F16 = HERE.parent / "ni26_friday_s1c_distill_20261118"


def paired(held, seed, smoke, subject, visit):
    if sha(HERE/"actor.sha256")!=ACTOR or sha(F16/"before_labels.sha256")!=F16_BANK:
        raise ValueError("F31 actor or signed F16 blind bank differs")
    source, streams, people = split(subject,visit,held)
    base,batches = authenticated(F16,F16/"checkpoints",held,seed,smoke,source,streams,people)
    _,source_tape=tape(batches,held,seed)
    tag=f"{'smoke_' if smoke else ''}{seed}_{held}"
    fixed=dict(held=held,seed=seed,epochs=base["epochs"],scale=base["scale"],data=DATA_SHA,origin=ORIGIN_SHA,
               runtime=RUNTIME,actor=ACTOR,source_rows=row_sha(source),source_people=people,batches=base["batches"],
               initial=base["initial"],parameters=347903,weights_origin=base["checkpoint"],steps=base["steps"],
               held_labels_opened=False,source_labels_only=True,rng_isolated=True,order_uses_task_labels=False,
               auxiliary_excluded=True,q_gradients_nonzero=0,torch=base["torch"],braindecode=base["braindecode"],**source_tape)
    result={}
    for kind in KINDS:
        path=HERE/"checkpoints"/f"{kind}_{tag}.pt"
        meta=json.loads(path.with_suffix(".json").read_text())
        expected=dict(fixed,kind=kind,checkpoint=sha(path),order_weight=0. if kind=="hard" else WEIGHT)
        taps=meta["filter_lengths"]
        if (any(meta[k]!=v for k,v in expected.items()) or "H100" not in meta["device"] or
                not np.isfinite([meta[k] for k in ("first_loss","last_loss","max_gradient")]).all() or
                not meta["max_gradient"]>0 or meta["state"]==meta["initial"] or
                len(taps)!=9 or any(n%2==0 for n in taps) or meta["fir_radius"]!=max(taps)//2 or
                not 0<meta["fir_radius"]<=120 or not 0<=meta["padding_error"]<=TOL):
            raise ValueError(f"original raw recipe or boundary controls differ for {kind}")
        if kind=="hard":
            if meta["state"]!=base["state"] or meta["checkpoint"]!=base["checkpoint"] or meta["aux_steps"]:
                raise ValueError("exact hard source anchor differs")
        else:
            signals=[meta[k] for k in ("head_gradient","body_gradient","pretext_loss","true_source_loss")]
            if (not np.isfinite(signals).all() or min(signals)<=0 or meta["aux_steps"]!=base["steps"] or
                    not 0<=meta["invariance_drift"]<=TOL or meta["auxiliary_parameters"]!=PARAMETERS or
                    meta["auxiliary_initial"]==meta["auxiliary_final"] or not 0<=meta["source_order_accuracy"]<=1):
                raise ValueError("auxiliary did not learn actual gain-independent waveform order")
        result[kind]=path,meta
    ordered,sham=[result[k][1] for k in ("ordered","sham")]
    if ordered["auxiliary_initial"]!=sham["auxiliary_initial"]:
        raise ValueError("order and sham head configurations differ")
    if smoke and not (ordered["true_source_loss"]<log(2) and ordered["true_source_loss"]<=sham["true_source_loss"]-.002):
        raise ValueError("real source pretext did not meet the registered activity threshold")
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
        evidence[kind]=dict(proof,checkpoint=meta["checkpoint"],state=meta["state"],tape=meta["tape"],
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
                raise ValueError("F31 hard held decisions differ from exact F16")
        path=HERE/"folds"/name
        with path.open("xb") as out:
            np.savez_compressed(out,index=rows,metadata=json.dumps(record,sort_keys=True),**decisions)
        with path.with_suffix(".sha256").open("x") as out:
            out.write(sha(path)+"\n")
    print(json.dumps(record,sort_keys=True),flush=True)
    print(f"PASS F31 held {held} seed {seed}",flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("held",choices=PEOPLE)
    cli.add_argument("seed",type=int,choices=(0,1))
    cli.add_argument("--smoke",action="store_true")
    args=cli.parse_args()
    run(args.held,args.seed,args.smoke)
