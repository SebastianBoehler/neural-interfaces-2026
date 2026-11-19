"""Replay and authenticate four frozen S1C models on untouched held EEG."""
import argparse
import json

import numpy as np
import torch

from anchors import DATA_SHA, ORIGIN_SHA, PEOPLE, sha
from calibration import ARMS, HERE, KINDS, MODES
from calibration_source import arm_fields, context, parent_roots
from held import frozen_member
from held_distill import replay
from replay_f13 import ORDER_SHA, ORDERS, origin_report, raw_inputs


ACTOR = "107d380fe6cb5e55532c3194986d7f0a5eb0705f48666fc8383fbedabfbaa3d2"


def paired(held, seed, smoke, subject, visit):
    if sha(HERE / "actor.sha256") != ACTOR:
        raise ValueError("F24 source manifest differs")
    refs, _, _, _, _, masks, _, selections, _, fixed = context(held, seed, smoke, subject, visit, True)
    tag = f"{'smoke_' if smoke else ''}{seed}_{held}"
    members = {}
    for kind in KINDS:
        path = HERE / "checkpoints" / f"{kind}_{tag}.pt"
        proof = json.loads(path.with_suffix(".json").read_text())
        expected = dict(fixed, **arm_fields(kind, selections, masks), checkpoint=sha(path), steps=refs["hard"]["steps"])
        if (any(proof[key] != v for key, v in expected.items()) or "H100" not in proof["device"] or
                not np.isfinite([proof[k] for k in ("first_loss", "last_loss", "max_gradient")]).all() or
                not 0 < proof["max_gradient"] or proof["state"] == proof["initial"]):
            raise ValueError(f"{kind} differs from the same source-only recipe")
        if kind in refs and any(proof[k] != refs[kind][k] for k in ("state", "checkpoint")):
            raise ValueError(f"{kind} differs from its sealed source parent")
        for key in ("cross_js", "within_js", "between_js"):
            if not np.isfinite(proof[key]) or proof[key] <= 0:
                raise ValueError("fixed clean source diagnostic did not run")
        if kind not in refs:
            signals = [proof[k] for k in ("aux_affine", "aux_spatial", "query_gradient_max", "aux_mean", "aux_movement")]
            if (not np.isfinite(signals).all() or min(signals) <= 0 or proof["donor_gradient_nonzero"] or
                    not 0 <= proof["native_delta"] <= 2e-5):
                raise ValueError(f"{kind} lost the detached auxiliary, query gradient or native readout")
        members[kind] = path, proof
    if len({m["state"] for _, m in members.values()}) != 4 or len({m["device"] for _, m in members.values()}) != 1:
        raise ValueError("duplicate or unmatched source actors")
    return members


def compare_archives(name, rows, decisions):
    for kind, root in parent_roots(True).items():
        with np.load(root / "folds" / name, allow_pickle=False) as prior:
            if not np.array_equal(prior["index"], rows) or any(
                    not np.array_equal(prior[f"{kind}_{m}"], decisions[f"{kind}_{m}"]) for m in MODES):
                raise ValueError(f"{kind} held decisions differ from their exact parent archive")


def run(held, seed, smoke):
    torch.set_num_threads(4)
    origin_report()
    x, subject, person, stream, perms = raw_inputs()
    visit = np.asarray([s.rsplit("|", 1)[-1] for s in stream])
    rows = np.flatnonzero(person == held)
    decisions, evidence = {}, {}
    for kind, (path, meta) in paired(held, seed, smoke, subject, visit).items():
        bank, proof = replay(*frozen_member(path, meta), x, stream, rows, perms, meta["scale"])
        for mode, value in zip(MODES, bank, strict=True):
            decisions[f"{kind}_{mode}"] = value
        evidence[kind] = dict(proof, checkpoint=meta["checkpoint"], state=meta["state"], context_sha=meta["context_sha"],
                              receipt=sha(path.with_suffix(".json")), jitter_sha=meta["jitter_sha"])
    if tuple(decisions) != ARMS or not all(p["clips"] for p in evidence.values()):
        raise ValueError("incomplete held arms or inactive bounded evaluation")
    receipt = dict(held=held, seed=seed, actor=ACTOR, scorer=sha(HERE / "score.sha256"), data=DATA_SHA,
                   origin=ORIGIN_SHA, orders=list(ORDERS), order_sha=ORDER_SHA, held_labels_opened=False,
                   evidence=evidence)
    if not smoke:
        name = f"{held}_s{seed}.npz"
        compare_archives(name, rows, decisions)
        path = HERE / "folds" / name
        with path.open("xb") as out:
            np.savez_compressed(out, index=rows, metadata=json.dumps(receipt, sort_keys=True), **decisions)
        with path.with_suffix(".sha256").open("x") as out:
            out.write(sha(path) + "\n")
    print(json.dumps(receipt, sort_keys=True), flush=True)
    print(f"PASS F24 held {held} seed {seed}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("held", choices=PEOPLE)
    parser.add_argument("seed", type=int, choices=(0, 1))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.held, args.seed, args.smoke)
