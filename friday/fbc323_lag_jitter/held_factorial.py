"""Replay and authenticate four frozen S1C models on untouched held EEG."""
import argparse
import json

import numpy as np
import torch

from anchors import DATA_SHA, ORIGIN_SHA, PEOPLE, sha
from factorial import ARMS, HERE, KINDS, MODES, arm_fields, common, context, operator, parent_roots, tag_for
from held_distill import replay
from replay_f13 import ORDER_SHA, ORDERS, origin_report, raw_inputs
from source_targets import normalizer
from temporal import ACTIVE_TOL, ANALYTIC_TOL, LIMIT, attach, permutation, trainable


ACTOR = "0ae228bed1dc7dd9d653dbe66ac51ca5d5a486e0054cd95e98eb3ecc431d276d"


def paired(held, seed, smoke, subject, visit):
    if sha(HERE / "actor.sha256") != ACTOR:
        raise ValueError("F23 source manifest differs")
    refs, rows, _, batches, _, masks, tape = context(held, seed, smoke, subject, visit, True)
    fixed = common(refs, rows, batches, masks, tape)
    tag = tag_for(held, seed, smoke)
    members = {}
    for kind in KINDS:
        path = HERE / "checkpoints" / f"{kind}_{tag}.pt"
        proof = json.loads(path.with_suffix(".json").read_text())
        expected = dict(fixed, **arm_fields(kind, masks), checkpoint=sha(path), steps=refs["hard"]["steps"])
        if (any(proof[key] != v for key, v in expected.items()) or "H100" not in proof["device"] or
                not np.isfinite([proof[k] for k in ("first_loss", "last_loss", "max_gradient")]).all() or
                not 0 < proof["max_gradient"] or proof["state"] == proof["initial"]):
            raise ValueError(f"{kind} differs from the same source-only recipe")
        if kind in refs and any(proof[k] != refs[kind][k] for k in ("state", "checkpoint")):
            raise ValueError(f"{kind} differs from its sealed source parent")
        if operator(kind) == "lag" and (
                not 0 <= proof["analytic_error"] <= ANALYTIC_TOL or
                not 0 < proof["coefficient_mean"] <= proof["coefficient_max"] <= LIMIT or
                not 0 < proof["nonzero_coefficients"] <= 288 or
                not np.isfinite([proof[k] for k in ("theta_gradient_max", "zero_delta", "swap_delta")]).all() or
                not proof["theta_gradient_max"] > 0 or not min(proof["zero_delta"], proof["swap_delta"]) > ACTIVE_TOL):
            raise ValueError(f"{kind} did not learn the real temporal pairing")
        members[kind] = path, proof
    if len({m["state"] for _, m in members.values()}) != 4 or len({m["device"] for _, m in members.values()}) != 1:
        raise ValueError("duplicate or unmatched source actors")
    return members


def frozen_member(path, meta):
    from encoder312 import fingerprint
    import model322

    model322.SEED = meta["seed"]
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if any(saved[k] != meta[k] for k in ("kind", "seed", "held", "scale")):
        raise ValueError("checkpoint does not match source fold identity")
    model = model322.Factor("session", False, "cpu")
    layer = attach(model, operator(meta["kind"]))
    model.load_state_dict(saved["state"], strict=True)
    model.eval()
    if trainable(model) != meta["parameters"] or fingerprint(model) != meta["state"]:
        raise ValueError("frozen state or parameter count differs")
    if layer is not None and (
            not np.array_equal(layer.pairs.numpy(), permutation("lag")) or
            int(layer.theta.detach().count_nonzero()) != meta["nonzero_coefficients"] or
            not np.isclose(float(layer.coefficients().detach().abs().max()),
                           meta["coefficient_max"], rtol=2e-6, atol=1e-8)):
        raise ValueError("frozen lag channel coefficients or pairing differ")
    return model, normalizer(model)


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
        evidence[kind] = dict(proof, checkpoint=meta["checkpoint"], state=meta["state"], pair=meta["pair"],
                              receipt=sha(path.with_suffix(".json")), jitter_tape=meta["jitter_tape"])
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
    print(f"PASS F23 held {held} seed {seed}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("held", choices=PEOPLE)
    parser.add_argument("seed", type=int, choices=(0, 1))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.held, args.seed, args.smoke)
