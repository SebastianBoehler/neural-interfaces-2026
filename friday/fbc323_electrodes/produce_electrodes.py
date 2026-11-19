"""Train exact hard S1C and two matched electrode-link nonlinearities."""
import argparse
from pathlib import Path

import numpy as np
import torch

from anchors import DATA_SHA, ORIGIN_SHA, PEOPLE, RUNTIME, row_sha, sha, split
from phase_source import F16_ACTOR, authenticated
from produce_distill import publish
from geometry_probe import diagnostics, fit, install
from source_probe import timing
from source_targets import normalizer
from electrodes import BASE, HERE, KINDS, MAX_RATIO, MAX_RESERVED, parameters, trainable
from topology import topology


F14 = HERE.parent / "friday_s1c_training_20261118"
F16 = HERE.parent / "friday_s1c_distill_20261118"


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
        x, subject, visit, channels = (z[k] for k in ("X", "subject", "session", "ch_names"))
    if x.shape != (2655, 41, 480) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError("wrong official waveform shape or values")
    graphs, geometry = topology(channels, False)
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
                 diagnostic_rows=row_sha(source[flat[0][:32]]), zero_identity=True, **geometry)
    tag = f"{'smoke_' if smoke else ''}{seed}_{held}"
    states = set()
    hard_ms = None
    for kind in KINDS:
        model = model322.Factor("session", False, "cuda")
        normalizer(model)
        if encoder312.fingerprint(model) != ref["initial"] or trainable(model) != BASE:
            raise ValueError("original initialization differs")
        layer = install(model, kind, graphs[kind], sample)
        stats = fit(model, layer, (x, y, mu, sd, weights, batches), ref)
        alt = graphs["scrambled" if kind == "physical" else "physical"]
        proof = {} if layer is None else diagnostics(model, layer, sample, alt)
        if smoke:
            ms = timing(model, sample)
            if kind == "hard":
                hard_ms = ms
            if not 0 < ms <= MAX_RATIO * hard_ms:
                raise ValueError(f"warm model-forward timing gate failed for {kind}: {ms:.3f} versus {hard_ms:.3f} ms")
            proof.update(warm_ms=ms, hard_ms=hard_ms)
        state = stats["state"]
        if state in states or encoder312.fingerprint(model) != state:
            raise ValueError("duplicate or mutable source model")
        states.add(state)
        reserved = torch.cuda.max_memory_reserved()
        if smoke and reserved > MAX_RESERVED:
            raise ValueError(f"H100 smoke exceeds preregistered CUDA reservation: {reserved}")
        output = HERE / f"{kind}_{tag}.pt"
        publish(output, model, dict(fixed, kind=kind, q_gradients_nonzero=0, peak_reserved=reserved,
                                    parameters=parameters(kind), graph=None if layer is None else row_sha(graphs[kind]),
                                    **stats, **proof))
        if kind == "hard" and sha(output) != ref["checkpoint"]:
            raise ValueError("hard F16 anchor differs byte-for-byte")
    print(f"H100 peak reserved {torch.cuda.max_memory_reserved()} bytes", flush=True)
    print(f"PASS F27 paired source {tag}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("held", choices=PEOPLE)
    parser.add_argument("seed", type=int, choices=(0, 1))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.held, args.seed, args.smoke)
