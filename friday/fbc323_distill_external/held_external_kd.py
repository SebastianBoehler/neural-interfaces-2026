"""Replay Scherer sessions using F16's raw-only head and the F15e delivery pins."""
import argparse
import json
from types import MethodType

import numpy as np
import torch

from anchors import KINDS, MODES, sha
from bounded_bn import normalize
from encoder312 import fingerprint
from external_inputs import DATA_SHA, ORDERS, PEOPLE, TOL, VISITS, ancestry, cohort, deliveries, delivery_receipt
from external_source import HERE
from held import frozen_member
from held_distill import raw_probability
from modes import probabilities
from receipts_external import ACTOR, paired
from replay_f13 import origin_report
from source_targets import normalizer


ARMS = tuple(f"s{seed}_{kind}_{mode}" for seed in (0, 1) for kind in KINDS for mode in MODES)


@torch.no_grad()
def replay(model, layer, x, scale, delivery, test):
    if normalizer(model) is not layer:
        raise ValueError("S1C source and held layer differ")
    model.skip = False
    before, native = fingerprint(model), layer.forward
    clips = []
    bounded = MethodType(lambda this, value: normalize(this, value, clips), layer)
    result = np.full((len(MODES), len(ORDERS), len(test)), -1, dtype=np.int8)
    first, resets, active, error = 0, 0, 0, 0.
    for (order, visit), rows in delivery.items():
        samples = x[rows] / scale
        cross = x[delivery[(order, VISITS[1 - VISITS.index(visit)])][:10]] / scale
        if not np.array_equal(raw_probability(model, samples[:10]), probabilities(model, samples[:10], "cpu")):
            raise ValueError("active raw-only output differs from the native zero-q control")
        for i, mode in enumerate((native, bounded)):
            layer.forward = mode
            clips.clear()
            post = raw_probability(model, samples)
            if not np.isfinite(post).all() or not np.allclose(post.sum(1), 1):
                raise ValueError("invalid held raw-only posterior")
            if i == 0:
                first_window = post[0].copy()
            else:
                if len(clips) != 1 or not np.array_equal(post[0], first_window):
                    raise ValueError("bounded causal history changed the actual first prediction")
                first += 1
                active += int(clips[0])
            prefix = raw_probability(model, samples[:10])
            if not np.allclose(prefix, post[:10], atol=TOL, rtol=TOL) or not np.array_equal(
                    prefix.argmax(1), post[:10].argmax(1)):
                raise ValueError("later windows changed a prefix")
            error = max(error, float(np.abs(prefix - post[:10]).max()))
            raw_probability(model, cross)
            if not np.array_equal(prefix, raw_probability(model, samples[:10])):
                raise ValueError("a different visit changed a restarted prefix")
            resets += 1
            result[i, ORDERS.index(order), np.searchsorted(test, rows)] = post.argmax(1)
        layer.forward = native
    if fingerprint(model) != before or (result < 0).any():
        raise ValueError("held features changed the actor or missed canonical rows")
    return result, dict(first_windows_exact=first, reset_checks=resets, clips=active, prefix_max_abs=error)


def run(ix, smoke):
    torch.set_num_threads(4)
    origin_report()
    ancestry()
    x, subject, _, _, stream = cohort(waveforms=True)
    person = PEOPLE[ix]
    rows = np.flatnonzero(subject == person)
    delivery = deliveries(stream, person)
    evidence, output = {}, {}
    for seed in (0,) if smoke else (0, 1):
        for kind, (p, meta) in paired(seed, smoke).items():
            preds, proof = replay(*frozen_member(p, meta), x, meta["scale"], delivery, rows)
            for mode, labels in zip(MODES, preds, strict=True):
                output[f"s{seed}_{kind}_{mode}"] = labels
            evidence[f"s{seed}_{kind}"] = dict(proof, checkpoint=meta["checkpoint"], state=meta["state"],
                                               receipt=sha(p.with_suffix(".json")), teacher_bank=meta["teacher_bank"],
                                               donor_plan=meta["donor_plan"])
    if tuple(output) != ARMS[:6 if smoke else 12] or not any(v["clips"] for v in evidence.values()):
        raise ValueError("missing inference or active bounded history")
    receipt = dict(person=person, actor=ACTOR, data=DATA_SHA, scorer=sha(HERE / "score.sha256"), held_labels_opened=False,
                   delivery=delivery_receipt(delivery), evidence=evidence)
    if not smoke:
        path = HERE / "banks" / f"person_{ix}.npz"
        with path.open("xb") as out:
            np.savez_compressed(out, index=rows, metadata=json.dumps(receipt, sort_keys=True), **output)
        with path.with_suffix(".sha256").open("x") as out:
            out.write(sha(path) + "\n")
    print(json.dumps(receipt, sort_keys=True), flush=True)
    print(f"PASS F16e Scherer {person}, {len(rows)} rows, {len(output)} arms", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("person_index", type=int, choices=range(len(PEOPLE)))
    cli.add_argument("--smoke", action="store_true")
    args = cli.parse_args()
    run(args.person_index, args.smoke)
