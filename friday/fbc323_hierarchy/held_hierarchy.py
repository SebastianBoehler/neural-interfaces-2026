"""Prove both exact flat controls and replay both frozen raw hierarchy variants."""
import argparse
import json

import numpy as np
import torch

from anchors import DATA_SHA, ORIGIN_SHA, PEOPLE, RUNTIME, row_sha, sha, split
from held import frozen_member
from held_distill import replay
from hierarchy_source import ARMS, F17_BANK, HERE, KINDS, MODES, controls
from replay_f13 import ORDER_SHA, ORDERS, origin_report, raw_inputs
from tree_replay import frozen_tree, replay_tree


ACTOR = "4770494bdeb0cef1fd767d2359759657a72c54032faf7c26c62e0493c702496e"
F17 = HERE.parent / "ni26_friday_s1c_phase_20261118"


def paired(held, seed, smoke, subject, visit):
    if sha(HERE / "actor.sha256") != ACTOR or sha(F17 / "before_labels.sha256") != F17_BANK:
        raise ValueError("F19 stage or F17 independent results lock differs")
    rows, stream, people = split(subject, visit, held)
    ref, _, _, masks, tape = controls(F17, F17 / "checkpoints", held, seed, smoke, rows, stream, people)
    stock = ref["hard"]
    fixed = dict(held=held, seed=seed, epochs=stock["epochs"], scale=stock["scale"], dataset=DATA_SHA,
                 origin=ORIGIN_SHA, runtime=RUNTIME, actor=ACTOR, batches=stock["batches"], flat_initial=stock["initial"],
                 source_rows=row_sha(rows), source_people=people, tape=tape, flat_origin=stock["checkpoint"],
                 jitter_origin=ref["jitter"]["checkpoint"], rng_isolated=True, q_gradients_nonzero=0, engineered_features=0,
                 train_rows_seen=stock["train_rows_seen"], planned_jitter_rows=stock["planned_rows"],
                 held_labels_opened=False, braindecode=stock["braindecode"], torch=stock["torch"], steps=len(masks))
    result = {}
    tag = f"{'smoke_' if smoke else ''}{seed}_{held}"
    for kind in KINDS:
        path = HERE / "checkpoints" / f"{kind}_{tag}.pt"
        m = json.loads(path.with_suffix(".json").read_text())
        tree = kind.startswith("hierarchy")
        expected = dict(fixed, kind=kind, checkpoint=sha(path), head_rows=2 if tree else 3,
                        jitter_rows=stock["planned_rows"] if kind.endswith("jitter") else 0)
        if (any(m[k] != v for k, v in expected.items()) or abs(m["parameters"] / stock["parameters"] - 1.) > .01 or
                m["state"] == m["initial"] or "H100" not in m["device"] or
                not np.isfinite([m[k] for k in ("first_loss", "last_loss", "max_gradient")]).all() or
                m["max_gradient"] <= 0):
            raise ValueError(f"unaligned genuine F19 source {kind}/{tag}")
        if tree:
            keys = ("motor_gradient", "conditional_gradient", "body_gradient", "motor_weight_delta", "conditional_weight_delta")
            if (not all(np.isfinite(m[k]) and m[k] > 0 for k in keys) or not 0 <= m["nll_error"] <= 2e-6 or
                    m["motor_conditional_gradient"] != 0 or not m["body_updated"]):
                raise ValueError("genuine end-to-end conditional objective was inactive")
        elif m["state"] != ref[kind]["state"] or m["parameters"] != stock["parameters"] or m["initial"] != stock["initial"]:
            raise ValueError("hard-CE or jitter-only exact ancestor state changed")
        result[kind] = path, m
    if (len({m["state"] for _, m in result.values()}) != 4 or len({m["device"] for _, m in result.values()}) != 1
            or result["hierarchy"][1]["initial"] != result["hierarchy_jitter"][1]["initial"]):
        raise ValueError("same-source factorial architecture or states differ")
    return result


def run(held, seed, smoke):
    torch.set_num_threads(4)
    origin_report()
    x, subject, person, stream, perms = raw_inputs()
    visit = np.asarray([s.rsplit("|", 1)[-1] for s in stream])
    rows = np.flatnonzero(person == held)
    decisions, evidence = {}, {}
    for kind, (path, meta) in paired(held, seed, smoke, subject, visit).items():
        if kind.startswith("hierarchy"):
            labels, proof = replay_tree(*frozen_tree(path, meta), x, stream, rows, perms, meta["scale"])
        else:
            labels, proof = replay(*frozen_member(path, meta), x, stream, rows, perms, meta["scale"])
        for mode, predictions in zip(MODES, labels, strict=True):
            decisions[f"{kind}_{mode}"] = predictions
        evidence[kind] = dict(proof, checkpoint=meta["checkpoint"], state=meta["state"], tape=meta["tape"],
                              receipt=sha(path.with_suffix(".json")))
    if tuple(decisions) != ARMS or not any(p["clips"] for p in evidence.values()):
        raise ValueError("incomplete factor replay or bounded work")
    result = dict(held=held, seed=seed, actor=ACTOR, scorer=sha(HERE / "score.sha256"), data=DATA_SHA,
                  origin=ORIGIN_SHA, orders=list(ORDERS), order_sha=ORDER_SHA, held_labels_opened=False, evidence=evidence)
    if not smoke:
        old = F17 / "folds" / f"{held}_s{seed}.npz"
        with np.load(old, allow_pickle=False) as z:
            if not np.array_equal(z["index"], rows) or any(not np.array_equal(z[f"{k}_{m}"], decisions[f"{k}_{m}"])
                    for k in ("hard", "jitter") for m in MODES):
                raise ValueError("both F17 held flat archives must match every decision")
        path = HERE / "folds" / old.name
        with path.open("xb") as f:
            np.savez_compressed(f, index=rows, metadata=json.dumps(result, sort_keys=True), **decisions)
        with path.with_suffix(".sha256").open("x") as f:
            f.write(sha(path) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)
    print(f"PASS F19 held {held} seed {seed}, four arms", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("held", choices=PEOPLE)
    cli.add_argument("seed", type=int, choices=(0, 1))
    cli.add_argument("--smoke", action="store_true")
    args = cli.parse_args()
    run(args.held, args.seed, args.smoke)
