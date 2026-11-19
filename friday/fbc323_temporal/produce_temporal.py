"""Train exact hard CE and two matched temporal-pooling S1C waveform models."""
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from anchors import DATA_SHA, ORIGIN_SHA, PEOPLE, RUNTIME, row_sha, sha, split
from phase_source import F16_ACTOR, authenticated
from produce_distill import publish
from source_probe import analytic, effects, timing
from source_targets import normalizer, raw_logits
from temporal import BASE_PARAMS, CHANNELS, HERE, KINDS, TIMING_CAP, attach, pair_sha, trainable


F14 = HERE.parent / "friday_s1c_training_20261118"
F16 = HERE.parent / "friday_s1c_distill_20261118"


@torch.no_grad()
def install(model, kind, sample):
    before = model.representation(sample)
    before_logits = raw_logits(model, sample)
    cpu, gpu = torch.random.get_rng_state(), torch.cuda.get_rng_state()
    layer = attach(model, kind)
    if (trainable(model) != BASE_PARAMS + (CHANNELS if layer is not None else 0) or
            not torch.equal(cpu, torch.random.get_rng_state()) or
            not torch.equal(gpu, torch.cuda.get_rng_state()) or
            not torch.equal(before, model.representation(sample)) or
            not torch.equal(before_logits, raw_logits(model, sample))):
        raise ValueError("zero coefficient failed exact representation, logit or RNG control")
    return layer


def fit_member(model, layer, kind, x, y, mu, sd, weights, batches, ref):
    import encoder312
    import model322

    q_gradients, theta_gradients = [], []
    head_hook = model.task.weight.register_hook(lambda g: q_gradients.append(g[:, -287:].count_nonzero()))
    theta_hook = None if layer is None else layer.theta.register_hook(lambda g: theta_gradients.append(g.abs().max()))
    try:
        stats = model322.train(model, x, y, mu, sd, weights, batches)
    finally:
        head_hook.remove()
        if theta_hook is not None:
            theta_hook.remove()
    n = ref["steps"]
    state = encoder312.fingerprint(model)
    if (stats["steps"] != n or len(q_gradients) != n or torch.stack(q_gradients).sum().item() or
            len(theta_gradients) != (n if layer is not None else 0) or
            state == ref["initial"] or (kind == "hard" and state != ref["state"])):
        raise ValueError(f"original fit controls or temporal gradient coverage differ for {kind}: {state}")
    if theta_gradients:
        magnitude = torch.stack(theta_gradients)
        if not magnitude.isfinite().all() or not (magnitude > 0).all():
            raise ValueError("missing actual trainable lag gradients")
        stats["theta_gradient_max"] = float(magnitude.max())
    return dict(stats, state=state, q_gradients_nonzero=0)


def run(held, seed, smoke):
    import braindecode
    import encoder312
    import model322
    from source_labels import read_training_labels

    if (sha(F16 / "actor.sha256") != F16_ACTOR or sha(F14 / "runtime.lock") != RUNTIME or
            sha(F14 / "original/source323.sha256") != ORIGIN_SHA or
            Path(model322.__file__).resolve() != F14 / "original/model322.py"):
        raise ValueError("original F16 source/runtime differs")
    device = torch.cuda.get_device_name()
    if "H100" not in device or braindecode.__version__ != "1.8.1":
        raise ValueError("requires the pinned H100 training platform")
    torch.set_num_threads(2)
    path = F14 / "proteus_windows.npz"
    if sha(path) != DATA_SHA:
        raise ValueError("original official waveform input differs")
    with np.load(path, allow_pickle=False) as z:
        x, subject, visit = (z[k] for k in ("X", "subject", "session"))
    if x.shape != (2655, 41, 480) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError("wrong official waveform shape or values")
    source, stream, people = split(subject, visit, held)
    ref, batches = authenticated(F16, F16, held, seed, smoke, source, stream, people)
    labels = read_training_labels(path, source, len(x))
    dat = dict(X=x[source], y=labels, stream=stream)
    x, y, scale, mu, sd, weights, prepared = model322.setup(dat, np.arange(len(source)), "cuda", epochs=ref["epochs"])
    flat = [b for epoch in batches for b in epoch]
    if scale != ref["scale"] or any(not np.array_equal(a, b) for a, b in zip(
            flat, (r for epoch in prepared for r in epoch), strict=True)):
        raise ValueError("original fit scale or source schedule changed")
    sample = x[flat[0][:32]]
    fixed = dict(held=held, seed=seed, epochs=ref["epochs"], scale=scale, data=DATA_SHA, origin=ORIGIN_SHA,
                 runtime=RUNTIME, weights_origin=ref["checkpoint"], initial=ref["initial"], batches=ref["batches"],
                 source_rows=row_sha(source), source_people=people, actor=sha(HERE / "actor.sha256"),
                 rng_isolated=True, held_labels_opened=False, device=device, torch=str(torch.__version__),
                 braindecode=braindecode.__version__, train_rows_seen=sum(map(len, flat)),
                 diagnostic_rows=row_sha(source[flat[0][:32]]), zero_identity=True)
    tag = f"{'smoke_' if smoke else ''}{seed}_{held}"
    states = set()
    hard_ms = None
    for kind in KINDS:
        model = model322.Factor("session", False, "cuda")
        normalizer(model)
        if encoder312.fingerprint(model) != ref["initial"] or trainable(model) != BASE_PARAMS:
            raise ValueError("original initialization differs")
        # Factor installs SessionBN after Encoder.eval(), so probes must use a separate eval copy.
        probe = deepcopy(model).eval()
        probe_layer = install(probe, kind, sample)
        proof = {} if probe_layer is None else {"analytic_error": analytic(probe, probe_layer, sample)}
        del probe, probe_layer
        if encoder312.fingerprint(model) != ref["initial"]:
            raise ValueError("source probe changed the pristine fit initialization")
        layer = attach(model, kind)
        stats = fit_member(model, layer, kind, x, y, mu, sd, weights, batches, ref)
        if smoke:
            ms = timing(model, sample)
            if kind == "hard":
                hard_ms = ms
            if not 0 < ms <= TIMING_CAP * hard_ms:
                raise ValueError(f"warm model-forward timing gate failed for {kind}: {ms:.3f} versus {hard_ms:.3f} ms")
            proof.update(warm_ms=ms, hard_ms=hard_ms)
        if layer is not None:
            proof.update(effects(model, layer, sample, kind))
        state = stats["state"]
        if state in states or encoder312.fingerprint(model) != state:
            raise ValueError("duplicate or mutable source model")
        states.add(state)
        publish(HERE / f"{kind}_{tag}.pt", model, dict(fixed, kind=kind, pair=pair_sha(kind),
                parameters=trainable(model), **stats, **proof))
    print(f"PASS F22 paired source {tag}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("held", choices=PEOPLE)
    parser.add_argument("seed", type=int, choices=(0, 1))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.held, args.seed, args.smoke)
