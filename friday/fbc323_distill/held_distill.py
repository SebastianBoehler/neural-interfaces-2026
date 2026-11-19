"""Replay F16 students without computing the fixed feature branch or reading labels."""
import argparse
import json
from types import MethodType

import numpy as np
import torch

from anchors import ARMS, DATA_SHA, HERE, KINDS, MODES, ORIGIN_SHA, PEOPLE, sha, split
from bounded_bn import normalize
from encoder312 import fingerprint
from held import frozen_member
from held_receipts import ACTOR_SHA, F15, source_receipts
from modes import probabilities
from replay_f13 import ORDER_SHA, ORDERS, origin_report, raw_inputs
from source_targets import TOL, normalizer, raw_logits


def raw_probability(model, x):
    tensor = torch.from_numpy(np.ascontiguousarray(x))
    return torch.softmax(raw_logits(model, tensor).double(), dim=1).cpu().numpy()


@torch.no_grad()
def replay(model, layer, x, stream, test, perms, scale):
    if normalizer(model) is not layer:
        raise ValueError("frozen evaluator differs from original SessionBN")
    model.skip = False
    before, native = fingerprint(model), layer.forward
    result = np.full((len(MODES), len(ORDERS), len(test)), -1, dtype=np.int8)
    counts = []
    bounded = MethodType(lambda this, value: normalize(this, value, counts), layer)
    checked = resets = active = 0
    prefix_max = 0.
    for j, order in enumerate(ORDERS):
        position = perms[order]
        for name in dict.fromkeys(stream[test]):
            rows = position[stream[position] == name]
            other = position[stream[position] != name][:10]
            sample, crossed = x[rows] / scale, x[other] / scale
            if not np.array_equal(raw_probability(model, sample[:10]), probabilities(model, sample[:10], "cpu")):
                raise ValueError("raw-only path lost F15 logical zero output identity")
            for m, call in enumerate((native, bounded)):
                layer.forward = call
                counts.clear()
                p = raw_probability(model, sample)
                if not np.isfinite(p).all() or not np.allclose(p.sum(1), 1):
                    raise ValueError("invalid held raw posterior")
                if m == 0:
                    first = p[0].copy()
                else:
                    if len(counts) != 1 or not np.array_equal(first, p[0]):
                        raise ValueError("bounded eval changed original first-window prediction")
                    active += int(counts[0])
                    checked += 1
                prefix = raw_probability(model, sample[:10])
                prefix_max = max(prefix_max, float(np.abs(p[:10] - prefix).max()))
                if not np.allclose(p[:10], prefix, atol=TOL, rtol=TOL) or not np.array_equal(
                        p[:10].argmax(1), prefix.argmax(1)):
                    raise ValueError("future raw windows changed the held prefix")
                raw_probability(model, crossed)
                if not np.array_equal(raw_probability(model, sample[:10]), prefix):
                    raise ValueError("raw evaluation retained state from a different visit")
                resets += 1
                result[m, j, np.searchsorted(test, rows)] = p.argmax(1)
            layer.forward = native
    if fingerprint(model) != before or (result < 0).any():
        raise ValueError("held coverage or frozen checkpoint identity failed")
    return result, dict(first_windows_exact=checked, resets=resets, clips=active, prefix_max_abs=prefix_max)


def run(held, seed, smoke):
    torch.set_num_threads(4)
    origin_report()
    x, subject, person, stream, perms = raw_inputs()
    test = np.flatnonzero(person == held)
    visit = np.asarray([s.rsplit("|", 1)[-1] for s in stream])
    source_rows, source_stream, source_people = split(subject, visit, held)
    evidence, decisions = {}, {}
    for kind, (path, meta) in source_receipts(held, seed, smoke, source_rows, source_stream, source_people).items():
        bank, proof = replay(*frozen_member(path, meta), x, stream, test, perms, meta["scale"])
        for mode, values in zip(MODES, bank, strict=True):
            decisions[f"{kind}_{mode}"] = values
        evidence[kind] = dict(proof, checkpoint=meta["checkpoint"], receipt=sha(path.with_suffix(".json")),
                              state=meta["state"], teacher_bank=meta["teacher_bank"], donor_plan=meta["donor_plan"])
    if tuple(decisions) != ARMS or set(evidence) != set(KINDS) or not any(v["clips"] for v in evidence.values()):
        raise ValueError("raw-only held matrix or bounded activity is incomplete")
    result = dict(held=held, seed=seed, actor=ACTOR_SHA, scorer=sha(HERE / "score.sha256"), data=DATA_SHA,
                  orders=list(ORDERS), order_sha=ORDER_SHA, origin=ORIGIN_SHA, held_labels_opened=False, evidence=evidence)
    if not smoke:
        with np.load(F15 / "folds" / f"{held}_s{seed}.npz", allow_pickle=False) as z:
            if not np.array_equal(z["index"], test) or any(not np.array_equal(
                    z[f"zero_{mode}_zero"], decisions[f"hard_{mode}"]) for mode in MODES):
                raise ValueError("recovered hard CE decoded differently from sealed F15 zero")
        path = HERE / "folds" / f"{held}_s{seed}.npz"
        with path.open("xb") as out:
            np.savez_compressed(out, index=test, metadata=json.dumps(result, sort_keys=True), **decisions)
        with path.with_suffix(".sha256").open("x") as f:
            f.write(sha(path) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)
    print(f"PASS F16 raw-only inference for {held} seed {seed}", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("held", choices=PEOPLE)
    cli.add_argument("seed", type=int, choices=(0, 1))
    cli.add_argument("--smoke", action="store_true")
    args = cli.parse_args()
    run(args.held, args.seed, args.smoke)
