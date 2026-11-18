"""Authenticate the matched source pairings and replay the frozen flat actors."""
import argparse
import json

import numpy as np
import torch

from anchors import DATA_SHA, ORIGIN_SHA, PEOPLE, RUNTIME, row_sha, sha, split
from held import frozen_member
from held_distill import replay
from donor_tape import make_tape, verify_tape
from latent_fit import MASS
from mix_source import ARMS, HERE, KINDS, MODES
from phase_source import F16_BANK, authenticated
from replay_f13 import ORDER_SHA, ORDERS, origin_report, raw_inputs


ACTOR = "26206380249683ff1e32e458e67b64789f7376feda57381ec227197d473f3038"
F16 = HERE.parent / "ni26_friday_s1c_distill_20261118"


def paired(held, seed, smoke, subject, visit, labels=None):
    if sha(HERE / "actor.sha256") != ACTOR or sha(F16 / "before_labels.sha256") != F16_BANK:
        raise ValueError("F20 or baked F16 blind source lock differs")
    source, stream, roster = split(subject, visit, held)
    ref, batches = authenticated(F16, F16 / "checkpoints", held, seed, smoke, source, stream, roster)
    tag = f"{'smoke_' if smoke else ''}{seed}_{held}"
    bank = HERE / "checkpoints" / f"donor_{tag}.npz"
    with np.load(bank, allow_pickle=False) as z:
        if not np.array_equal(z["index"], source):
            raise ValueError("label-blind training rows do not match the authenticated fit")
        steps, positions = z["batch"], z["positions"]
    _, _, exposure = verify_tape(batches, subject[source], stream, steps, positions, labels=labels)
    if labels is not None:
        replay_step, replay_slots = make_tape(batches, subject[source], stream, labels, held, seed)
        if not np.array_equal(steps, replay_step) or not np.array_equal(positions, replay_slots):
            raise ValueError("fixed same-task source tape failed its independent label check")
    fixed = dict(held=held, seed=seed, epochs=ref["epochs"], scale=ref["scale"], dataset=DATA_SHA, origin=ORIGIN_SHA,
                 runtime=RUNTIME, weights_origin=ref["checkpoint"], initial=ref["initial"], parameters=347903,
                 batches=ref["batches"], source_rows=row_sha(source), source_people=roster, actor=ACTOR,
                 donor_bank=sha(bank), held_labels_opened=False, rng_isolated=True, mass=MASS,
                 q_gradients_nonzero=0, steps=ref["steps"], braindecode=ref["braindecode"], torch=ref["torch"], **exposure)
    result = {}
    for kind in KINDS:
        path = HERE / "checkpoints" / f"{kind}_{tag}.pt"
        meta = json.loads(path.with_suffix(".json").read_text())
        required = dict(fixed, kind=kind, checkpoint=sha(path), mixed_rows=exposure["matched_rows"] if kind != "hard" else 0)
        if (any(meta[key] != v for key, v in required.items()) or "H100" not in meta["device"] or
                meta["mechanism_quartets"] != 64 or not np.isfinite(
                    [meta[k] for k in ("within_cosine_distance", "across_cosine_distance", "across_minus_within_cosine")]).all() or
                not np.isfinite([meta[key] for key in ("first_loss", "last_loss", "max_gradient")]).all() or
                meta["max_gradient"] <= 0 or meta["initial"] == meta["state"] or
                kind == "hard" and meta["state"] != ref["state"]):
            raise ValueError(f"incorrect matched source augmentation or hard reference for {kind}/{tag}")
        if kind != "hard" and (min(meta["anchor_gradient"], meta["donor_gradient"]) <= 0 or
                not np.isclose(meta["anchor_gradient"] / 3, meta["donor_gradient"], atol=2e-7, rtol=2e-6)):
            raise ValueError("actual NLL gradient failed the selected endpoints and fixed mass")
        result[kind] = path, meta
    if len({m["state"] for _, m in result.values()}) != 3 or len({m["device"] for _, m in result.values()}) != 1:
        raise ValueError("duplicate or unmatched source models")
    return result


def run(held, seed, smoke):
    torch.set_num_threads(4)
    origin_report()
    x, subject, person, stream, perms = raw_inputs()
    visit = np.asarray([s.rsplit("|", 1)[-1] for s in stream])
    rows = np.flatnonzero(person == held)
    decisions, evidence = {}, {}
    for kind, (path, meta) in paired(held, seed, smoke, subject, visit).items():
        labels, proof = replay(*frozen_member(path, meta), x, stream, rows, perms, meta["scale"])
        for mode, budget in zip(MODES, labels, strict=True):
            decisions[f"{kind}_{mode}"] = budget
        evidence[kind] = dict(proof, checkpoint=meta["checkpoint"], state=meta["state"], tape=meta["tape"], donor_bank=meta["donor_bank"],
                              receipt=sha(path.with_suffix(".json")))
    if tuple(decisions) != ARMS or not any(p["clips"] for p in evidence.values()):
        raise ValueError("missing held arm or bounded work")
    receipt = dict(held=held, seed=seed, actor=ACTOR, scorer=sha(HERE / "score.sha256"), data=DATA_SHA,
                   origin=ORIGIN_SHA, orders=list(ORDERS), order_sha=ORDER_SHA, held_labels_opened=False,
                   evidence=evidence)
    if not smoke:
        original = F16 / "folds" / f"{held}_s{seed}.npz"
        with np.load(original, allow_pickle=False) as z:
            if not np.array_equal(z["index"], rows) or any(not np.array_equal(
                    z[f"hard_{m}"], decisions[f"hard_{m}"]) for m in MODES):
                raise ValueError("F20 hard decisions differ from sealed F16")
        path = HERE / "folds" / original.name
        with path.open("xb") as out:
            np.savez_compressed(out, index=rows, metadata=json.dumps(receipt, sort_keys=True), **decisions)
        with path.with_suffix(".sha256").open("x") as out:
            out.write(sha(path) + "\n")
    print(json.dumps(receipt, sort_keys=True), flush=True)
    print(f"PASS F20 held {held} seed {seed}", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("held", choices=PEOPLE)
    cli.add_argument("seed", type=int, choices=(0, 1))
    cli.add_argument("--smoke", action="store_true")
    a = cli.parse_args()
    run(a.held, a.seed, a.smoke)
