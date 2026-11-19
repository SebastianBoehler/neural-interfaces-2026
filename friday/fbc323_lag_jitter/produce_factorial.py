"""Fit all four raw S1C arms using the unchanged F17 and F22 native operators."""
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from anchors import DATA_SHA, ORIGIN_SHA, PEOPLE, RUNTIME, sha
from factorial import HERE, KINDS, arm_fields, common, context, is_jittered, operator, tag_for
from perturb import Feed, perturb, random_tape
from produce_distill import publish
from produce_temporal import fit_member, install
from source_probe import analytic, effects
from source_targets import normalizer
from temporal import BASE_PARAMS, attach, trainable


F14 = HERE.parent / "friday_s1c_training_20261118"


def jitter_inputs(x, batches, rows, held, seed, directions, masks, tape):
    cpu, gpu = torch.random.get_rng_state(), torch.cuda.get_rng_state()
    signs, direction, selected, reproduced = random_tape(batches, len(rows), held, seed)
    if (reproduced != tape or not np.array_equal(direction, directions) or
            any(not np.array_equal(a, b) for a, b in zip(selected, masks, strict=True))):
        raise ValueError("exact F17 sign, direction or selection tape differs")
    first = [r for ep in batches for r in ep][:3]
    sampled = np.unique(np.concatenate([r[m] for r, m in zip(first, masks[:3], strict=True)]))[:8]
    jitter, _, guards = perturb(x, directions, signs, sampled)
    if (not torch.equal(cpu, torch.random.get_rng_state()) or not torch.equal(gpu, torch.cuda.get_rng_state()) or
            sum(int(m.sum()) for m in masks) * 4 != sum(map(len, masks))):
        raise ValueError("jitter either consumed RNG or missed the exact quarter source exposure")
    return jitter, guards


def run(held, seed, smoke):
    from encoder312 import fingerprint
    import braindecode
    import model322
    from source_labels import read_training_labels

    if (sha(F14 / "runtime.lock") != RUNTIME or sha(F14 / "original/source323.sha256") != ORIGIN_SHA or
            Path(model322.__file__).resolve() != F14 / "original/model322.py"):
        raise ValueError("original source runtime differs")
    device = torch.cuda.get_device_name()
    if "H100" not in device or braindecode.__version__ != "1.8.1":
        raise ValueError("exact H100 training runtime required")
    torch.set_num_threads(2)
    path = F14 / "proteus_windows.npz"
    if sha(path) != DATA_SHA:
        raise ValueError("official waveform input differs")
    with np.load(path, allow_pickle=False) as z:
        x, subject, visit = (z[k] for k in ("X", "subject", "session"))
    if x.shape != (2655, 41, 480) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError("invalid raw waveform input")
    refs, rows, stream, batches, directions, masks, tape = context(held, seed, smoke, subject, visit, False)
    base = refs["hard"]
    labels = read_training_labels(path, rows, len(x))
    dat = dict(X=x[rows], y=labels, stream=stream)
    x, y, scale, mu, sd, weights, prepared = model322.setup(dat, np.arange(len(rows)), "cuda", epochs=base["epochs"])
    flat = [r for ep in batches for r in ep]
    if scale != base["scale"] or any(not np.array_equal(a, b) for a, b in zip(
            flat, (r for ep in prepared for r in ep), strict=True)):
        raise ValueError("original source scale or schedule differs")
    jitter, guards = jitter_inputs(x, batches, rows, held, seed, directions, masks, tape)
    fixed = dict(common(refs, rows, batches, masks, tape), device=device, **guards)
    tag = tag_for(held, seed, smoke)
    sample = x[flat[0][:32]]
    states = set()
    for kind in KINDS:
        model = model322.Factor("session", False, "cuda")
        normalizer(model)
        if fingerprint(model) != base["initial"] or trainable(model) != BASE_PARAMS:
            raise ValueError("model did not start from exact zero-feature S1C")
        geometry = operator(kind)
        probe = deepcopy(model).eval()
        probe_layer = install(probe, geometry, sample)
        proof = {} if probe_layer is None else dict(analytic_error=analytic(probe, probe_layer, sample))
        del probe, probe_layer
        if fingerprint(model) != base["initial"]:
            raise ValueError("source probe mutated the fit initialization")
        layer = attach(model, geometry)
        native = model.forward
        feed = Feed(jitter, batches, masks) if is_jittered(kind) else None
        if feed is not None:
            model.forward = feed.install(model, native, x)
        try:
            stats = fit_member(model, layer, kind, x, y, mu, sd, weights, batches, base)
        finally:
            model.forward = native
        fields = arm_fields(kind, masks)
        if feed is not None and feed.calls != fields["feed_calls"]:
            raise ValueError("F17 source hook did not execute on all scheduled batches")
        if layer is not None:
            proof.update(effects(model, layer, sample, "lag"))
        state = stats["state"]
        if (state in states or fingerprint(model) != state or trainable(model) != fields["parameters"] or
                kind in refs and state != refs[kind]["state"]):
            raise ValueError(f"mutable, duplicate or mismatched exact {kind} source state")
        states.add(state)
        output = HERE / f"{kind}_{tag}.pt"
        receipt = dict(fixed, **fields, **proof, **{k: v for k, v in stats.items() if k != "q_gradients_nonzero"})
        publish(output, model, receipt)
        if kind in refs and sha(output) != refs[kind]["checkpoint"]:
            raise ValueError(f"{kind} did not reproduce the exact parent checkpoint bytes")
    print(f"PASS F23 paired source {tag}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("held", choices=PEOPLE)
    parser.add_argument("seed", type=int, choices=(0, 1))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.held, args.seed, args.smoke)
