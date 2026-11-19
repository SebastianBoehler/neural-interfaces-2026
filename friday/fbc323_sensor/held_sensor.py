"""Authenticate the sensor plan and replay the original raw-only F16 held actor."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from anchors import DATA_SHA, ORIGIN_SHA, PEOPLE, RUNTIME, row_sha, sha, split
from held import frozen_member
from held_distill import replay
from phase_source import F16_BANK, MODES, authenticated
from withholding import MONTAGE, plan
from replay_f13 import DATA, ORDER_SHA, ORDERS, origin_report, raw_inputs


HERE = Path(__file__).resolve().parent
ACTOR = "e7fd28145f16d25d8e1fecd7d47bcac0a3820e854d0e10b18d55b265e9ecf2a9"
KINDS = ("hard", "local", "dispersed")
ARMS = tuple(f"{k}_{m}" for k in KINDS for m in MODES)
F16 = HERE.parent / "ni26_friday_s1c_distill_20261118"


def paired(held, seed, smoke, x, channels, subject, visit):
    if sha(HERE / "actor.sha256") != ACTOR or sha(F16 / "before_labels.sha256") != F16_BANK:
        raise ValueError("F18 or baked F16 blind source lock differs")
    source, stream, roster = split(subject, visit, held)
    ref, batches = authenticated(F16, F16 / "checkpoints", held, seed, smoke, source, stream, roster)
    nearby, dispersed, masks, tape, evidence = plan(x[source], channels, batches, held, seed)
    tag = f"{'smoke_' if smoke else ''}{seed}_{held}"
    bank = HERE / "checkpoints" / f"mask_{tag}.npz"
    with np.load(bank, allow_pickle=False) as z:
        if not all(np.array_equal(z[key], value) for key, value in (
                ("index", source), ("nearby", nearby), ("dispersed", dispersed))):
            raise ValueError("source waveform and canonical geometry did not reproduce the sealed label-blind mask plan")
    replaced, total = sum(int(m.sum()) for m in masks), sum(map(len, masks))
    fixed = dict(held=held, seed=seed, epochs=ref["epochs"], scale=ref["scale"], dataset=DATA_SHA, origin=ORIGIN_SHA,
                 runtime=RUNTIME, weights_origin=ref["checkpoint"], initial=ref["initial"], parameters=347903,
                 batches=ref["batches"], source_rows=row_sha(source), source_people=roster, actor=ACTOR,
                 train_rows_seen=total, planned_rows=replaced, tape=tape, mask_bank=sha(bank), montage=MONTAGE, held_labels_opened=False, rng_isolated=True,
                 q_gradients_nonzero=0, steps=ref["steps"], braindecode=ref["braindecode"], torch=ref["torch"])
    result = {}
    for kind in KINDS:
        path = HERE / "checkpoints" / f"{kind}_{tag}.pt"
        meta = json.loads(path.with_suffix(".json").read_text())
        required = dict(fixed, kind=kind, checkpoint=sha(path), perturbed_rows=replaced if kind != "hard" else 0)
        if (any(meta[key] != v for key, v in required.items()) or "H100" not in meta["device"] or
                any(abs(meta[k] - v) > 1e-12 for k, v in evidence.items()) or
                not np.isfinite([meta[key] for key in ("first_loss", "last_loss", "max_gradient")]).all() or
                meta["max_gradient"] <= 0 or meta["initial"] == meta["state"] or
                kind == "hard" and meta["state"] != ref["state"]):
            raise ValueError(f"incorrect matched source augmentation or hard reference for {kind}/{tag}")
        result[kind] = path, meta
    if len({m["state"] for _, m in result.values()}) != 3 or len({m["device"] for _, m in result.values()}) != 1:
        raise ValueError("duplicate or unmatched source models")
    return result


def run(held, seed, smoke):
    torch.set_num_threads(4)
    origin_report()
    x, subject, person, stream, perms = raw_inputs()
    visit = np.asarray([s.rsplit("|", 1)[-1] for s in stream])
    with np.load(DATA, allow_pickle=False) as z:
        channels = z["ch_names"]
    rows = np.flatnonzero(person == held)
    decisions, evidence = {}, {}
    for kind, (path, meta) in paired(held, seed, smoke, x, channels, subject, visit).items():
        labels, proof = replay(*frozen_member(path, meta), x, stream, rows, perms, meta["scale"])
        for mode, budget in zip(MODES, labels, strict=True):
            decisions[f"{kind}_{mode}"] = budget
        evidence[kind] = dict(proof, checkpoint=meta["checkpoint"], state=meta["state"], tape=meta["tape"], mask_bank=meta["mask_bank"],
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
                raise ValueError("F18 hard decisions differ from sealed F16")
        path = HERE / "folds" / original.name
        with path.open("xb") as out:
            np.savez_compressed(out, index=rows, metadata=json.dumps(receipt, sort_keys=True), **decisions)
        with path.with_suffix(".sha256").open("x") as out:
            out.write(sha(path) + "\n")
    print(json.dumps(receipt, sort_keys=True), flush=True)
    print(f"PASS F18 held {held} seed {seed}", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("held", choices=PEOPLE)
    cli.add_argument("seed", type=int, choices=(0, 1))
    cli.add_argument("--smoke", action="store_true")
    a = cli.parse_args()
    run(a.held, a.seed, a.smoke)
