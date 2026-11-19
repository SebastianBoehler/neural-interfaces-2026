"""Verify fit exclusion and replay each frozen temporal decoder on untouched EEG."""
import argparse
import json

import numpy as np
import torch

from anchors import DATA_SHA, ORIGIN_SHA, PEOPLE, RUNTIME, row_sha, sha, split
from held_distill import replay
from phase_source import F16_BANK, authenticated
from replay_f13 import ORDER_SHA, ORDERS, origin_report, raw_inputs
from source_targets import normalizer
from modulation import ACTIVE, ARMS, BANDS, DILATIONS, HERE, KINDS, MAX_RATIO, MAX_RESERVED, MODES
from modulation import architecture, attach, trainable


ACTOR = "589b9a212cafdb0367f9f34bb840db5ff95c98ccfeea3bf77e268aa3ddb2e95c"
F16 = HERE.parent / "ni26_friday_s1c_distill_20261118"


def paired(held, seed, smoke, subject, visit):
    if sha(HERE / "actor.sha256") != ACTOR or sha(F16 / "before_labels.sha256") != F16_BANK:
        raise ValueError("pinned F25 source or F16 blind lock differs")
    source, stream, people = split(subject, visit, held)
    ref, batches = authenticated(F16, F16 / "checkpoints", held, seed, smoke, source, stream, people)
    flat = [b for ep in batches for b in ep]
    fixed = dict(held=held, seed=seed, epochs=ref["epochs"], scale=ref["scale"], data=DATA_SHA, origin=ORIGIN_SHA,
                 runtime=RUNTIME, weights_origin=ref["checkpoint"], initial=ref["initial"], batches=ref["batches"],
                 source_rows=row_sha(source), source_people=people, actor=ACTOR, rng_isolated=True,
                 held_labels_opened=False, train_rows_seen=sum(map(len, flat)), diagnostic_rows=row_sha(source[flat[0][:32]]),
                 zero_identity=True, q_gradients_nonzero=0, steps=ref["steps"], braindecode=ref["braindecode"],
                 torch=ref["torch"])
    result = {}
    tag = f"{'smoke_' if smoke else ''}{seed}_{held}"
    for kind in KINDS:
        path = HERE / "checkpoints" / f"{kind}_{tag}.pt"
        proof = json.loads(path.with_suffix(".json").read_text())
        expected = dict(fixed, kind=kind, checkpoint=sha(path), **architecture(kind))
        if (any(proof[key] != v for key, v in expected.items()) or "H100" not in proof["device"] or
                not np.isfinite([proof[key] for key in ("first_loss", "last_loss", "max_gradient")]).all() or
                proof["max_gradient"] <= 0 or proof["initial"] == proof["state"]):
            raise ValueError(f"original source scope or S1C geometry differs for {kind}/{tag}")
        if not 0 < proof["peak_reserved"]:
            raise ValueError("invalid CUDA memory receipt")
        if kind == "hard":
            if proof["state"] != ref["state"] or proof["checkpoint"] != ref["checkpoint"]:
                raise ValueError("F25 hard state differs byte-for-byte from exact F16")
        else:
            effects = [proof[k] for k in ("band_grad", "tap_grad", "identity_delta", "dilation_delta",
                                          "cross_band", "edge_tap")]
            if (not np.isfinite(effects).all() or min(effects) <= ACTIVE or
                    not .75 <= proof["gain_min"] < proof["gain_max"] <= 1.25):
                raise ValueError(f"learned source modulation did not activate for {kind}/{tag}")
        result[kind] = path, proof
    all_meta = [m for _, m in result.values()]
    if len({m["state"] for m in all_meta}) != len(KINDS) or len({m["device"] for m in all_meta}) != 1:
        raise ValueError("duplicate or unmatched fitted arms")
    if smoke:
        hard_ms = result["hard"][1]["warm_ms"]
        if any(m["hard_ms"] != hard_ms or not 0 < m["warm_ms"] <= MAX_RATIO * hard_ms or m["peak_reserved"] > MAX_RESERVED for m in all_meta):
            raise ValueError("paired warm source forward timing gate changed")
    return result


def frozen_member(path, meta):
    import encoder312
    import model322

    kind = meta["kind"]
    model322.SEED = meta["seed"]
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if any(payload[key] != meta[key] for key in ("kind", "seed", "held", "scale")):
        raise ValueError("checkpoint and fit identity differ")
    model = model322.Factor("session", False, "cpu")
    layer = attach(model, kind)
    model.load_state_dict(payload["state"], strict=True)
    model.eval()
    if trainable(model) != meta["parameters"] or encoder312.fingerprint(model) != meta["state"]:
        raise ValueError("frozen parameter count or F25 state differs")
    if layer is not None:
        bands = layer.bands.detach().reshape(-1, BANDS, BANDS)
        diagonal = torch.eye(BANDS, dtype=torch.bool)
        cross = float(bands[:, ~diagonal].abs().max())
        edge = float(torch.cat([w.detach()[..., (0, 1, 3, 4)].flatten() for w in layer.taps]).abs().max())
        if (tuple(meta["dilations"]) != layer.dilations or layer.dilations != DILATIONS[kind] or
                cross != meta["cross_band"] or edge != meta["edge_tap"]):
            raise ValueError("frozen cross-band and temporal taps differ from their proof")
    return model, normalizer(model)


def run(held, seed, smoke):
    torch.set_num_threads(4)
    origin_report()
    x, subject, person, stream, perms = raw_inputs()
    visit = np.asarray([s.rsplit("|", 1)[-1] for s in stream])
    rows = np.flatnonzero(person == held)
    decisions, evidence = {}, {}
    for kind, (path, meta) in paired(held, seed, smoke, subject, visit).items():
        labels, proof = replay(*frozen_member(path, meta), x, stream, rows, perms, meta["scale"])
        for mode, result in zip(MODES, labels, strict=True):
            decisions[f"{kind}_{mode}"] = result
        evidence[kind] = dict(proof, checkpoint=meta["checkpoint"], state=meta["state"], dilations=meta["dilations"],
                              receipt=sha(path.with_suffix(".json")))
    if tuple(decisions) != ARMS or not any(p["clips"] for p in evidence.values()):
        raise ValueError("missing held arm or bounded work")
    receipt = dict(held=held, seed=seed, actor=ACTOR, scorer=sha(HERE / "score.sha256"), data=DATA_SHA,
                   origin=ORIGIN_SHA, orders=list(ORDERS), order_sha=ORDER_SHA, held_labels_opened=False, evidence=evidence)
    if not smoke:
        original = F16 / "folds" / f"{held}_s{seed}.npz"
        with np.load(original, allow_pickle=False) as z:
            if not np.array_equal(z["index"], rows) or any(not np.array_equal(
                    z[f"hard_{m}"], decisions[f"hard_{m}"]) for m in MODES):
                raise ValueError("F25 hard decisions differ from sealed F16")
        path = HERE / "folds" / original.name
        with path.open("xb") as out:
            np.savez_compressed(out, index=rows, metadata=json.dumps(receipt, sort_keys=True), **decisions)
        with path.with_suffix(".sha256").open("x") as out:
            out.write(sha(path) + "\n")
    print(json.dumps(receipt, sort_keys=True), flush=True)
    print(f"PASS F25 held {held} seed {seed}", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("held", choices=PEOPLE)
    cli.add_argument("seed", type=int, choices=(0, 1))
    cli.add_argument("--smoke", action="store_true")
    args = cli.parse_args()
    run(args.held, args.seed, args.smoke)
