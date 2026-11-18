"""Replay all three S1C fits with ordinary and bounded inference, without held labels."""
import argparse
import hashlib
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from bounded_bn import normalize
from replay_f13 import DATA_SHA, ORDER_SHA, ORDERS, ORIGIN_SHA, PEOPLE, origin_report, raw_inputs, row_sha, session_batches, sha


HERE = Path(__file__).resolve().parent
KINDS = ("original", "causal", "bounded")
ARMS = tuple(f"{kind}_{mode}" for kind in KINDS for mode in ("ordinary", "bounded"))
TOL = 2e-5


def paired(held, seed, subject, people, stream, smoke):
    import model322

    fit = np.flatnonzero(people != held)
    epochs = 4 if smoke else 40
    model322.SEED = seed
    episodes = model322.batches(stream[fit], epochs)
    batch = hashlib.sha256(b"".join(fit[b].astype("<i8").tobytes() for ep in episodes for b in ep)).hexdigest()
    ancestry = dict(held=held, seed=seed, epochs=epochs, dataset=DATA_SHA, origin=ORIGIN_SHA,
                    actor=sha(HERE / "actor.sha256"), runtime=sha(HERE / "runtime.lock"), batches=batch,
                    source_rows=row_sha(fit), source_people=sorted(set(subject[fit])), held_labels_opened=False,
                    steps=sum(map(len, episodes)), braindecode="1.8.1")
    pairs = []
    for kind in KINDS:
        key = f"{'smoke_' if smoke else ''}{kind}_{seed}_{held}"
        checkpoint = HERE / "checkpoints" / f"{key}.pt"
        meta = json.loads(checkpoint.with_suffix(".json").read_text())
        expected = dict(ancestry, kind=kind, checkpoint=sha(checkpoint))
        if any(meta[k] != value for k, value in expected.items()) or "H100" not in meta["device"]:
            raise ValueError(f"source-only checkpoint provenance differs for {key}")
        if bool(meta["active_training_clips"]) != (kind == "bounded") or not all(
                np.isfinite(meta[k]) for k in ("scale", "first_loss", "last_loss", "max_gradient")):
            raise ValueError(f"invalid source activity for {key}")
        pairs.append((checkpoint, meta))
    for key in ("scale", "initial", "parameters", "device", "torch"):
        if len({m[key] for _, m in pairs}) != 1:
            raise ValueError(f"unpaired source training {key}")
    if len({m["state"] for _, m in pairs}) != len(KINDS):
        raise ValueError("duplicated trained state")
    return pairs


def frozen_member(checkpoint, meta):
    import encoder312
    import model322
    from session_bn import SessionBN

    model322.SEED = meta["seed"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if any(payload[key] != meta[key] for key in ("kind", "seed", "held", "scale")):
        raise ValueError("checkpoint does not identify the authenticated training fold")
    model = model322.Factor("session", True, "cpu")
    model.load_state_dict(payload["state"], strict=True)
    model.eval()
    layers = [layer for layer in model.body.modules() if type(layer) is SessionBN]
    if len(layers) != 1 or layers[0].k != 20 or encoder312.fingerprint(model) != meta["state"]:
        raise ValueError("fitted state or original S1C SessionBN path differs")
    return model, layers[0]


@torch.no_grad()
def evaluate_one(model, layer, x, stream, test, perms, scale):
    from encoder312 import fingerprint
    from modes import probabilities

    before, forward = fingerprint(model), layer.forward
    result = np.full((2, len(ORDERS), len(test), 3), np.nan)
    activity, prefix_error, first_windows = 0, 0.0, 0
    counts = []
    bounded = MethodType(lambda self, inp: normalize(self, inp, counts), layer)
    for order, pos, xx, other in session_batches(x, stream, test, scale, perms):
        prior = probabilities(model, xx, "cpu")
        layer.forward = bounded
        counts.clear()
        current = probabilities(model, xx, "cpu")
        if len(counts) != 1 or not np.array_equal(prior[0], current[0]):
            raise ValueError("eval modes differ on the first actual window")
        activity += int(counts[0])
        prefix = probabilities(model, xx[:10], "cpu")
        discrepancy = float(np.abs(current[:len(prefix)] - prefix).max())
        prefix_error = max(prefix_error, discrepancy)
        if not np.allclose(current[:len(prefix)], prefix, rtol=TOL, atol=TOL):
            raise ValueError("future inputs changed a bounded held prefix")
        if len(other):
            probabilities(model, other, "cpu")
            if not np.array_equal(probabilities(model, xx[:10], "cpu"), prefix):
                raise ValueError("held state crosses the session boundary")
        layer.forward = forward
        if not np.allclose(prior[:len(prefix)], probabilities(model, xx[:10], "cpu"), rtol=TOL, atol=TOL):
            raise ValueError("future inputs changed an ordinary held prefix")
        result[:, ORDERS.index(order), pos] = np.stack((prior, current))
        first_windows += 1
    if fingerprint(model) != before or not np.isfinite(result).all() or not np.allclose(result.sum(3), 1):
        raise ValueError("frozen state or output alignment changed during held inference")
    return result.argmax(3).astype(np.int8), dict(active_eval_clips=activity, prefix_max_abs=prefix_error,
                                                first_windows_exact=first_windows)


def run(held, seed, smoke):
    torch.set_num_threads(4)
    origin_report()
    x, subject, people, stream, perms = raw_inputs()
    pairs = paired(held, seed, subject, people, stream, smoke)
    test = np.flatnonzero(people == held)
    decisions, details = {}, {}
    for checkpoint, meta in pairs:
        kind = meta["kind"]
        out, guards = evaluate_one(*frozen_member(checkpoint, meta), x, stream, test, perms, meta["scale"])
        decisions[f"{kind}_ordinary"], decisions[f"{kind}_bounded"] = out
        details[kind] = dict(guards, checkpoint=meta["checkpoint"], source_state=meta["state"],
                             source_rows=meta["source_rows"], receipt=sha(checkpoint.with_suffix(".json")))
    if not any(m["active_eval_clips"] for m in details.values()):
        raise ValueError("held eval control exercised no clipped history")
    receipt = dict(held=held, seed=seed, orders=list(ORDERS), order_sha=ORDER_SHA, dataset=DATA_SHA, origin=ORIGIN_SHA,
                   scorer=sha(HERE / "score.sha256"), actor=sha(HERE / "actor.sha256"),
                   held_labels_opened=False, guards=details)
    if not smoke:
        output = HERE / "folds" / f"{held}_s{seed}.npz"
        with output.open("xb") as handle:
            np.savez_compressed(handle, index=test, metadata=json.dumps(receipt, sort_keys=True), **decisions)
        print(f"{sha(output)}  {output.name}", flush=True)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    print("PASS F14 independent held inference", flush=True)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("held", choices=PEOPLE)
    args.add_argument("seed", choices=(0, 1), type=int)
    args.add_argument("--smoke", action="store_true")
    cli = args.parse_args()
    run(cli.held, cli.seed, cli.smoke)
