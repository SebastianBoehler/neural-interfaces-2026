"""Independently derive held bounded F36 pupils and verify the F-C22h hard anchor."""
import argparse
import json

import numpy as np
import torch

from anchors import PEOPLE, sha
from encoder312 import fingerprint
from held import frozen_member
from held_distill import raw_probability
from pins_f36c import ACTOR, BLIND, DATA, DELIVERY, DEVICE, F36, ORDERS, PUPIL_ARMS, REF_BLIND, REFERENCE, RESULT, SCORER, WORK
from pupils import do_order
from replay_f13 import ORDERS as ORIGINAL_ORDERS, origin_report, raw_inputs
from pair_checks import paired


def pinned_folds(name):
    manifests = ((F36 / "actor.sha256", ACTOR), (F36 / "score.sha256", SCORER),
                 (F36 / "before_labels.sha256", BLIND), (REFERENCE / "before-labels.json", REF_BLIND),
                 (F36 / "whole_results.json", RESULT))
    if any(sha(p) != digest for p,digest in manifests):
        raise ValueError("immutable F36 or F-C22h manifest differs")
    f36 = dict(line.split()[::-1] for line in manifests[2][0].read_text().splitlines())
    reference = json.loads(manifests[3][0].read_text())
    digests = [reference["orders"][order]["folds"][name] for order in ORDERS]
    old, hard = F36 / "folds" / name, REFERENCE / "pupils" / name
    if (sha(old) != f36[f"folds/{name}"] or any(sha(hard) != d for d in digests)
            or reference["held_labels_opened"] or len(f36) != 20):
        raise ValueError("original independently sealed F36 or F-C22h fold bytes differ")
    return old, hard


@torch.no_grad()
def run(held, seed, smoke):
    torch.set_num_threads(2)
    origin_report()
    x, subject, person, stream, perms = raw_inputs()
    visit = np.array([name.rsplit("|",1)[-1] for name in stream])
    rows = np.flatnonzero(person == held)
    name = f"{held}_s{seed}.npz"
    old, hard = pinned_folds(name)
    cases = paired(held,seed,False,subject,visit)
    observed, receipts = {}, {}
    with np.load(old,allow_pickle=False) as signed, np.load(hard,allow_pickle=False) as reference:
        for bank in (signed,reference):
            np.testing.assert_array_equal(bank["index"],rows)
        lock = json.loads(str(signed["metadata"]))
        native = json.loads(str(reference["metadata"]))
        expected = dict(held=held,seed=seed,actor=ACTOR,scorer=SCORER,data=DATA,order_sha=DELIVERY,
                        orders=list(ORIGINAL_ORDERS),device=DEVICE,held_labels_opened=False)
        if any(lock[k] != v for k,v in expected.items()) or set(cases) != set(PUPIL_ARMS):
            raise ValueError("sealed F36 held fold scope differs")
        if native["arms"]["hard"]["checkpoint"] != cases["hard"][1]["checkpoint"]:
            raise ValueError("same-device F-C22h hard fit bytes differ")
        for kind,(path,meta) in cases.items():
            proof = lock["evidence"][kind]
            evidence = dict(checkpoint=meta["checkpoint"],state=meta["state"],receipt=sha(path.with_suffix(".json")),tape=meta["tape"])
            if any(proof[k] != v for k,v in evidence.items()):
                raise ValueError("replayed actor or source receipt differs from signed F36")
            model, layer = frozen_member(path,meta)
            model.skip = False
            count = 0
            for order in ORDERS:
                p, prefix, clips = do_order(model,layer,x,stream,rows,perms[order],meta["scale"])
                for visit_id in dict.fromkeys(stream[rows]):
                    delivery = perms[order][stream[perms[order]] == visit_id]
                    ordinary = raw_probability(model,x[delivery] / meta["scale"])
                    np.testing.assert_array_equal(ordinary[0],p[np.searchsorted(rows,delivery[0])])
                    count += 1
                np.testing.assert_array_equal(p.argmax(1),signed[f"{kind}_bounded"][ORIGINAL_ORDERS.index(order)])
                if kind == "hard":
                    np.testing.assert_array_equal(p,reference[f"hard|{order}"])
                observed[f"{kind}|{order}"] = p
                observed[f"{kind}|{order}|proof"] = np.array([prefix,clips])
            if fingerprint(model) != meta["state"] or count != 2*len(set(stream[rows])):
                raise ValueError("frozen actor or causal prefix scope differs")
            receipts[kind] = dict(evidence,exact_first_windows=count,exact_interleaved_resets=count)
    dest = WORK / ("smoke" if smoke else "pupils") / name
    provenance = dict(person=held,seed=seed,data=DATA,order_sha=DELIVERY,actor=ACTOR,device=DEVICE,
                      source=sha(WORK/"pupil.sha256"),signed_f36=sha(old),hard_reference=sha(hard),
                      hard_bitwise=True,held_labels_opened=False,arms=receipts)
    with dest.open("xb") as f:
        np.savez_compressed(f,index=rows,metadata=json.dumps(provenance,sort_keys=True),**observed)
    digest = sha(dest)
    stamp = dest.with_suffix(".sha256")
    with stamp.open("x") as f:
        f.write(digest+"\n")
    dest.chmod(0o444)
    stamp.chmod(0o444)
    print("PASS F36c independent pupils",held,seed,digest,flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("held",choices=PEOPLE)
    cli.add_argument("seed",type=int,choices=(0,1))
    cli.add_argument("--smoke",action="store_true")
    a = cli.parse_args()
    run(a.held,a.seed,a.smoke)
