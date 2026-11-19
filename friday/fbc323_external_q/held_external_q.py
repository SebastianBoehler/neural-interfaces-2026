"""Use the same F15e weights for active and zero fixed-feature Scherer replay."""
import argparse
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from bounded_bn import normalize
from encoder312 import fingerprint
from evaluate import probability
from external_inputs import DATA_SHA, ORDERS, PEOPLE, TOL, VISITS, ancestry, cohort, deliveries, delivery_receipt
from held import frozen_member
from held_distill import raw_probability
from held_external import paired
from modes import probabilities
from replay_f13 import origin_report, sha


HERE = Path(__file__).resolve().parent
F15E = HERE.parent / "ni26_friday_s1c_skip_external_20261118"
F15E_BANK = "ff25cdfee588ad9085ca85d9ac97bf26218621aca5a0b454a9813e2caaff2884"
KINDS = ("original", "dropout")
MODES = ("ordinary", "bounded")
FEATURES = ("on", "zero")
ARMS = tuple(f"s{s}_{kind}_{m}_{q}" for s in (0, 1) for kind in KINDS for m in MODES for q in FEATURES)


def old_lock():
    path = F15E / "before_labels.json"
    if sha(path) != F15E_BANK:
        raise ValueError("immutable F15e blind evidence differs")
    return json.loads(path.read_text())


def sources(seeds, history):
    result = {}
    for seed in seeds:
        available = paired(seed, smoke=False)
        for kind, (p, meta) in available.items():
            if kind not in KINDS:
                continue
            key = f"s{seed}_{kind}"
            proof = dict(checkpoint=meta["checkpoint"], state=meta["state"], receipt=sha(p.with_suffix(".json")))
            if history["source"][key] != proof:
                raise ValueError("F15e checkpoint differs from its blind lock")
            result[key] = p, meta, proof
    if len(result) != len(seeds) * len(KINDS):
        raise ValueError("a fixed all-source S1C checkpoint is missing")
    return result


@torch.no_grad()
def replay(model, layer, x, scale, delivery, test):
    before, native = fingerprint(model), layer.forward
    clips = []
    bounded = MethodType(lambda this, value: normalize(this, value, clips), layer)
    result = np.full((len(MODES), len(FEATURES), len(ORDERS), len(test)), -1, dtype=np.int8)
    first, resets, bounded_work = 0, 0, 0
    error, effect = 0., 0.
    for (order, visit), rows in delivery.items():
        samples = x[rows] / scale
        cross = x[delivery[(order, VISITS[1 - VISITS.index(visit)])][:10]] / scale
        for i, normalizer in enumerate((native, bounded)):
            layer.forward = normalizer
            clips.clear()
            post = probability(model, samples, FEATURES)
            if not np.isfinite(post).all() or not np.allclose(post.sum(2), 1):
                raise ValueError("invalid same-weight paired probabilities")
            if i == 0:
                first_window = post[:, 0].copy()
            else:
                if len(clips) != 1 or not np.array_equal(first_window, post[:, 0]):
                    raise ValueError("one fixed-feature mode changed first-window normalizer control")
                first += len(FEATURES)
                bounded_work += int(clips[0])
            effect = max(effect, float(np.abs(post[0] - post[1]).max()))
            prefix = probability(model, samples[:10], FEATURES)
            if not np.allclose(prefix, post[:, :10], atol=TOL, rtol=TOL) or not np.array_equal(
                    prefix.argmax(2), post[:, :10].argmax(2)):
                raise ValueError("later windows invalidated a fixed-feature prefix")
            for q, values in zip(FEATURES, prefix, strict=True):
                model.skip = q == "on"
                if not np.array_equal(values, probabilities(model, samples[:10], "cpu")):
                    raise ValueError("shared representation differs from the native active or zero path")
            model.skip = True
            if not np.array_equal(prefix[1], raw_probability(model, samples[:10])):
                raise ValueError("zero-q replay differs from the path that omits feature computation")
            error = max(error, float(np.abs(prefix - post[:, :10]).max()))
            probability(model, cross, FEATURES)
            if not np.array_equal(prefix, probability(model, samples[:10], FEATURES)):
                raise ValueError("a different visit changed the restarted prefix")
            resets += len(FEATURES)
            result[i, :, ORDERS.index(order)][:, np.searchsorted(test, rows)] = post.argmax(2)
        layer.forward = native
    if fingerprint(model) != before or (result < 0).any() or effect <= 1e-4:
        raise ValueError("frozen weights, decision coverage or feature activity failed")
    return result, dict(first=first, resets=resets, clipped=bounded_work, prefix_max=error, q_effect=effect)


def run(ix, smoke):
    torch.set_num_threads(4)
    origin_report()
    ancestry()
    history = old_lock()
    x, subject, _, _, stream = cohort(waveforms=True)
    person = PEOPLE[ix]
    rows = np.flatnonzero(subject == person)
    delivery = deliveries(stream, person)
    bank = F15E / "banks" / f"person_{ix}.npz"
    archive = sha(bank)
    if archive != history["banks"][bank.name]:
        raise ValueError("archived F15e person predictions differ")
    out, evidence = {}, {}
    with np.load(bank, allow_pickle=False) as original:
        if not np.array_equal(original["index"], rows):
            raise ValueError("archived F15e canonical row alignment differs")
        for key, (p, meta, proof) in sources((0,) if smoke else (0, 1), history).items():
            post, checks = replay(*frozen_member(p, meta), x, meta["scale"], delivery, rows)
            for i, mode in enumerate(MODES):
                if not np.array_equal(post[i, 0], original[f"{key}_{mode}"]):
                    raise ValueError("full-q decisions differ from the full authenticated F15e person bank")
                for j, q in enumerate(FEATURES):
                    out[f"{key}_{mode}_{q}"] = post[i, j]
            evidence[key] = dict(proof, **checks, full_q_archive_matches=len(rows) * len(ORDERS) * len(MODES))
    if tuple(out) != ARMS[:8 if smoke else 16] or not any(p["clipped"] for p in evidence.values()):
        raise ValueError("paired replay is missing a q endpoint or bounded activity")
    result = dict(person=person, data=DATA_SHA, scorer=sha(HERE / "score.sha256"), archive=archive,
                  held_labels_opened=False, delivery=delivery_receipt(delivery), evidence=evidence)
    if not smoke:
        path = HERE / "banks" / f"person_{ix}.npz"
        with path.open("xb") as handle:
            np.savez_compressed(handle, index=rows, metadata=json.dumps(result, sort_keys=True), **out)
        with path.with_suffix(".sha256").open("x") as handle:
            handle.write(sha(path) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)
    print(f"PASS F15eq Scherer {person}, {len(rows)} rows, {len(out)} endpoints", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("person_index", type=int, choices=range(len(PEOPLE)))
    cli.add_argument("--smoke", action="store_true")
    args = cli.parse_args()
    run(args.person_index, args.smoke)
