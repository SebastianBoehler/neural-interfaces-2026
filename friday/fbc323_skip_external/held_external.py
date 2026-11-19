"""Replay full-q Scherer decision banks independently of the H100 trainer."""
import argparse
import hashlib
import json
from types import MethodType

import numpy as np
import torch

from bounded_bn import normalize
from encoder312 import fingerprint
from evaluate import planned_masks, probability
from external_inputs import (ACTOR_SHA, ARMS, DATA_SHA, HERE, KINDS, MODES, ORDERS, PEOPLE, SOURCE, SOURCE_SHA, TOL,
                             VISITS, ancestry, cohort, deliveries, delivery_receipt)
from held import frozen_member
from modes import probabilities
from replay_f13 import ORIGIN_SHA, origin_report, row_sha, sha


RUNTIME_SHA = "da21360c9a85d7561eee8f0f68367b545797641a69e3726354275cbf404ec30e"


def paired(seed, smoke):
    import model322

    with np.load(SOURCE, allow_pickle=False) as z:
        subject, visit = z["subject"], z["session"]
    stream = np.char.add(np.char.add(subject, "|"), visit)
    rows = np.arange(2655)
    model322.SEED = seed
    epochs = 4 if smoke else 40
    batches = [b for epoch in model322.batches(stream, epochs) for b in epoch]
    schedule = hashlib.sha256(b"".join(rows[b].astype("<i8").tobytes() for b in batches)).hexdigest()
    count = sum(map(len, batches))
    fixed = dict(held="all_source", seed=seed, epochs=epochs, dataset=SOURCE_SHA, origin=ORIGIN_SHA,
                 actor=ACTOR_SHA, runtime=RUNTIME_SHA, batches=schedule, source_rows=row_sha(rows),
                 source_people=sorted(set(subject)), held_labels_opened=False, steps=len(batches), parameters=347903,
                 train_rows_seen=count, planned_mask_rows=count // 4, mask_plan=planned_masks(batches, "P00", seed),
                 mask_rng_isolated=True, braindecode="1.8.1", q_gradient_active_steps=len(batches))
    if (len(fixed["source_people"]) != 10 or set(fixed["source_people"]) & set(PEOPLE) or
            len(subject) != len(rows) or not all(p.startswith("Dreyer2026Proteus/") for p in subject)):
        raise ValueError("wrong source roster or external cohort leakage")
    result = {}
    for kind in KINDS:
        path = HERE / "checkpoints" / f"{'smoke_' if smoke else ''}{kind}_{seed}_all_source.pt"
        meta = json.loads(path.with_suffix(".json").read_text())
        expected = dict(fixed, kind=kind, checkpoint=sha(path), active_mask_rows=count // 4 if kind == "dropout" else 0)
        if (any(meta[key] != value for key, value in expected.items()) or "H100" not in meta["device"] or
                meta["state"] == meta["initial"] or not meta["max_q_gradient"] > 0 or not meta["scale"] > 0 or
                not np.isfinite([meta[k] for k in ("scale", "max_q_gradient", "first_loss", "last_loss")]).all()):
            raise ValueError(f"wrong all-source fit receipt for seed {seed}: {kind}")
        result[kind] = path, meta
    for key in ("scale", "initial", "device", "torch"):
        if len({meta[key] for _, meta in result.values()}) != 1:
            raise ValueError(f"unpaired source training setting {key}")
    if len({meta["state"] for _, meta in result.values()}) != len(KINDS):
        raise ValueError("duplicate source states")
    return result


def full_q(model, x):
    return probability(model, x, ("on",))[0]


@torch.no_grad()
def replay(model, layer, x, scale, delivery, test):
    before, native = fingerprint(model), layer.forward
    clips = []
    bounded = MethodType(lambda self, rows: normalize(self, rows, clips), layer)
    result = np.full((len(MODES), len(ORDERS), len(test)), -1, dtype=np.int8)
    first = resets = active = 0
    discrepancy = 0.0
    for (order, visit), rows in delivery.items():
        xy = x[rows] / scale
        interleaved = x[delivery[(order, VISITS[1 - VISITS.index(visit)])][:10]] / scale
        if not np.array_equal(full_q(model, xy[:10]), probabilities(model, xy[:10], "cpu")):
            raise ValueError("held q-on representation differs from original Factor")
        for i, forward in enumerate((native, bounded)):
            layer.forward = forward
            clips.clear()
            post = full_q(model, xy)
            if not np.isfinite(post).all() or not np.allclose(post.sum(1), 1):
                raise ValueError("invalid full-q Scherer posterior")
            if i == 0:
                baseline_first = post[0].copy()
            else:
                if len(clips) != 1 or not np.array_equal(post[0], baseline_first):
                    raise ValueError("bounded history changed the actual first window")
                first += 1
                active += int(clips[0])
            prefix = full_q(model, xy[:10])
            difference = float(np.abs(post[:10] - prefix).max())
            if not np.allclose(post[:10], prefix, atol=TOL, rtol=TOL) or not np.array_equal(
                    post[:10].argmax(1), prefix.argmax(1)):
                raise ValueError("future windows changed a held prefix")
            full_q(model, interleaved)
            if not np.array_equal(full_q(model, xy[:10]), prefix):
                raise ValueError("another visit changed a restarted prefix")
            resets += 1
            discrepancy = max(discrepancy, difference)
            result[i, ORDERS.index(order), np.searchsorted(test, rows)] = post.argmax(1)
        layer.forward = native
    if fingerprint(model) != before or (result < 0).any():
        raise ValueError("incomplete held decisions or changed fitted weights")
    return result, dict(first_windows_exact=first, reset_checks=resets, active_eval_clips=active,
                        prefix_max_abs=discrepancy)


def run(person, smoke):
    torch.set_num_threads(4)
    origin_report()
    ancestry()
    x, subject, _, _, stream = cohort(waveforms=True)
    test = np.flatnonzero(subject == person)
    delivery = deliveries(stream, person)
    decisions, evidence = {}, {}
    for seed in (0,) if smoke else (0, 1):
        for kind, (path, meta) in paired(seed, smoke).items():
            got, guards = replay(*frozen_member(path, meta), x, meta["scale"], delivery, test)
            for mode, labels in zip(MODES, got, strict=True):
                decisions[f"s{seed}_{kind}_{mode}"] = labels
            evidence[f"s{seed}_{kind}"] = dict(guards, checkpoint=meta["checkpoint"], state=meta["state"],
                                               receipt=sha(path.with_suffix(".json")))
    if tuple(decisions) != ARMS[:6 if smoke else 12] or not any(g["active_eval_clips"] for g in evidence.values()):
        raise ValueError("incomplete or inactive external replay")
    receipt = dict(person=person, actor=ACTOR_SHA, scorer=sha(HERE / "score.sha256"), dataset=DATA_SHA,
                   delivery=delivery_receipt(delivery), held_labels_opened=False, evidence=evidence)
    if not smoke:
        output = HERE / "banks" / f"person_{PEOPLE.index(person)}.npz"
        with output.open("xb") as handle:
            np.savez_compressed(handle, index=test, metadata=json.dumps(receipt, sort_keys=True), **decisions)
        with output.with_suffix(".sha256").open("x") as handle:
            handle.write(sha(output) + "\n")
    print(json.dumps(receipt, sort_keys=True), flush=True)
    print(f"PASS F15e external replay {person}, {len(test)} rows and {len(decisions)} frozen arms", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("person_index", choices=range(len(PEOPLE)), type=int)
    cli.add_argument("--smoke", action="store_true")
    args = cli.parse_args()
    run(PEOPLE[args.person_index], args.smoke)
