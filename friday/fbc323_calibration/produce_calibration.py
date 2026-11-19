"""Fit native raw S1C with source-only within/cross-session calibration targets."""
import argparse
from pathlib import Path

import numpy as np
import torch

from anchors import DATA_SHA, ORIGIN_SHA, PEOPLE, RUNTIME, sha
from calibration import HERE, KINDS
from calibration_source import arm_fields, context
from perturb import Feed, perturb, random_tape
from produce_distill import publish
from source_targets import normalizer
from train_context import diagnose, train


F14 = HERE.parent / "friday_s1c_training_20261118"


def jitter_inputs(x, batches, rows, held, seed, directions, masks, fixed):
    cpu, gpu = torch.random.get_rng_state(), torch.cuda.get_rng_state()
    signs, direction, selected, digest = random_tape(batches, len(rows), held, seed)
    if (digest != fixed["jitter_sha"] or not np.array_equal(direction, directions) or
            any(not np.array_equal(a, b) for a, b in zip(selected, masks, strict=True))):
        raise ValueError("source jitter tape differs from F17")
    flat = [r for ep in batches for r in ep]
    sampled = np.unique(np.concatenate([r[m] for r, m in zip(flat[:3], masks[:3], strict=True)]))[:8]
    jitter, _, guards = perturb(x, directions, signs, sampled)
    if (not torch.equal(cpu, torch.random.get_rng_state()) or not torch.equal(gpu, torch.cuda.get_rng_state()) or
            sum(int(m.sum()) for m in masks) * 4 != sum(map(len, masks))):
        raise ValueError("jitter either consumed RNG or missed the exact source exposure")
    return jitter, guards


def run(held, seed, smoke):
    import braindecode
    import model322
    from encoder312 import fingerprint
    from source_labels import read_training_labels

    path = F14 / "proteus_windows.npz"
    if (sha(F14 / "runtime.lock") != RUNTIME or sha(F14 / "original/source323.sha256") != ORIGIN_SHA or
            Path(model322.__file__).resolve() != F14 / "original/model322.py" or sha(path) != DATA_SHA):
        raise ValueError("original waveform input or runtime differs")
    device = torch.cuda.get_device_name()
    if "H100" not in device or braindecode.__version__ != "1.8.1":
        raise ValueError("exact H100 training runtime required")
    torch.set_num_threads(2)
    with np.load(path, allow_pickle=False) as z:
        x, subject, visit = (z[k] for k in ("X", "subject", "session"))
    if x.shape != (2655, 41, 480) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError("invalid raw waveform input")
    refs, rows, streams, batches, directions, masks, flat, selections, picked, fixed = context(
        held, seed, smoke, subject, visit, False)
    labels = read_training_labels(path, rows, len(x))
    dat = dict(X=x[rows], y=labels, stream=streams)
    x, y, scale, mu, sd, weights, prepared = model322.setup(dat, np.arange(len(rows)), "cuda", epochs=fixed["epochs"])
    if scale != fixed["scale"] or any(not np.array_equal(a, b) for a, b in zip(
            flat, (r for ep in prepared for r in ep), strict=True)):
        raise ValueError("original source scale or schedule differs")
    jitter, guards = jitter_inputs(x, batches, rows, held, seed, directions, masks, fixed)
    tag = f"{'smoke_' if smoke else ''}{seed}_{held}"
    states = set()
    for kind in KINDS:
        model = model322.Factor("session", False, "cuda")
        normalizer(model)
        if fingerprint(model) != fixed["initial"] or sum(p.numel() for p in model.parameters()) != 347903:
            raise ValueError("native zero-feature S1C source initial state differs")
        native = model.forward
        feed = None if kind == "hard" else Feed(jitter, batches, masks)
        if feed is not None:
            model.forward = feed.install(model, native, x)
        activity = []
        hook = model.task.weight.register_hook(lambda grad: activity.append(grad[:, -287:].count_nonzero()))
        try:
            if kind in refs:
                stats = model322.train(model, x, y, mu, sd, weights, batches)
            else:
                stats = train(model, x, y, mu, sd, weights, flat, selections, seed, kind)
        finally:
            hook.remove()
            model.forward = native
        fields = arm_fields(kind, selections, masks)
        n = refs["hard"]["steps"]
        # The separate auxiliary gradient probe targets only affine/spatial tensors, not the head.
        if (stats["steps"] != n or len(activity) != n or torch.stack(activity).sum().item() or
                feed is not None and feed.calls != fields["feed_calls"] or
                kind not in refs and any(stats[k] != fields[k] for k in ("aux_steps", "aux_queries"))):
            raise ValueError("original work, jitter or same-context exposure differs")
        state = fingerprint(model)
        diagnostic = diagnose(model, x, flat, picked)
        if (state in states or fingerprint(model) != state or state == fixed["initial"] or
                kind in refs and state != refs[kind]["state"]):
            raise ValueError(f"mutable, duplicate or mismatched exact {kind} source state")
        states.add(state)
        output = HERE / f"{kind}_{tag}.pt"
        publish(output, model, dict(fixed, **(fields | stats), **guards, **diagnostic, device=device, state=state))
        if kind in refs and sha(output) != refs[kind]["checkpoint"]:
            raise ValueError(f"{kind} did not reproduce the exact parent checkpoint bytes")
    print(f"PASS F24 paired source {tag}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("held", choices=PEOPLE)
    parser.add_argument("seed", type=int, choices=(0, 1))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.held, args.seed, args.smoke)
