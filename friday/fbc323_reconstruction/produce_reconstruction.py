"""Train clean, reconstructed CE, and matched reconstruction-consistency S1C."""
import argparse
from pathlib import Path

import numpy as np
import torch

from anchors import DATA_SHA, ORIGIN_SHA, PEOPLE, RUNTIME, row_sha, sha, split
from pair_fit import source_js, train
from phase_source import F16_ACTOR, authenticated
from produce_distill import publish
from reconstruction import CONSISTENCY, HERE, KINDS, MAP, apply_maps, geometry, view_tape
from source_targets import normalizer


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
        raise ValueError("original F16 training runtime differs")
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name() or braindecode.__version__ != "1.8.1":
        raise ValueError("requires the same source H100 runtime")
    torch.set_num_threads(2)
    path = F14 / "proteus_windows.npz"
    if sha(path) != DATA_SHA:
        raise ValueError("original official waveform input differs")
    with np.load(path, allow_pickle=False) as z:
        x, subject, visit, channels = (z[k] for k in ("X", "subject", "session", "ch_names"))
    if x.shape != (2655, 41, 480) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError("wrong raw waveform shape or values")
    source, stream, people = split(subject, visit, held)
    ref, batches = authenticated(F16, F16, held, seed, smoke, source, stream, people)
    rng = torch.random.get_rng_state(), torch.cuda.get_rng_state()
    keep, maps = geometry(channels)
    codes, tape = view_tape(batches, held, seed)
    y = read_training_labels(path, source, len(x))
    dat = dict(X=x[source], y=y, stream=stream)
    x, y, scale, mu, sd, weights, prepared = model322.setup(dat, np.arange(len(source)), "cuda", epochs=ref["epochs"])
    flat = [b for ep in batches for b in ep]
    if scale != ref["scale"] or any(not np.array_equal(a, b) for a, b in zip(
            flat, (r for ep in prepared for r in ep), strict=True)):
        raise ValueError("clean source schedule or scale changed")
    views, gap = apply_maps(x, keep, maps)
    if not torch.equal(rng[0], torch.random.get_rng_state()) or not torch.equal(rng[1], torch.cuda.get_rng_state()):
        raise ValueError("fixed geometry and schedule changed model RNG")
    picked = np.concatenate([rows[code >= 0] for rows, code in zip(flat, codes, strict=True)])[:64]
    seen = sum(map(len, flat))
    fixed = dict(held=held, seed=seed, epochs=ref["epochs"], scale=scale, dataset=DATA_SHA, origin=ORIGIN_SHA,
                runtime=RUNTIME, weights_origin=ref["checkpoint"], initial=ref["initial"], parameters=347903,
                batches=ref["batches"], source_rows=row_sha(source), source_people=people, map=MAP, tape=tape,
                actor=sha(HERE / "actor.sha256"), rng_isolated=True, held_labels_opened=False,
                device=torch.cuda.get_device_name(), torch=str(torch.__version__), braindecode=braindecode.__version__,
                reconstructed_change=gap, train_rows_seen=seen, diagnostic_rows=row_sha(source[picked]))
    tag = f"{'smoke_' if smoke else ''}{seed}_{held}"
    states = set()
    for kind in KINDS:
        model = model322.Factor("session", False, "cuda")
        normalizer(model)
        if (model.skip or encoder312.fingerprint(model) != ref["initial"] or
                sum(p.numel() for p in model.parameters() if p.requires_grad) != 347903):
            raise ValueError("source actor differs from original zero-feature architecture or init")
        skip_grads = []
        hook = model.task.weight.register_hook(lambda g: skip_grads.append(torch.count_nonzero(g[:, -287:])))
        strength = CONSISTENCY if kind == "consistent" else 0.
        if kind == "hard":
            fit = dict(model322.train(model, x, y, mu, sd, weights, batches), view_counts=[0, 0])
        else:
            fit = train(model, x, y, mu, sd, weights, batches, codes, views, strength, seed)
        hook.remove()
        state = encoder312.fingerprint(model)
        if (fit["steps"] != ref["steps"] or len(skip_grads) != ref["steps"] or torch.stack(skip_grads).sum().item() or
                state == ref["initial"] or state in states or kind == "hard" and state != ref["state"] or
                fit["view_counts"] != ([0, 0] if kind == "hard" else [seen // 8] * 2)):
            raise ValueError("source consistency strength, actual view exposures or F16 hard state differs")
        divergence = source_js(model, stream, views, picked)
        if encoder312.fingerprint(model) != state:
            raise ValueError("unmixed source observation changed fitted weights")
        states.add(state)
        publish(HERE / f"{kind}_{tag}.pt", model, dict(fixed, kind=kind, state=state, strength=strength,
                q_gradients_nonzero=0, source_js=divergence, **fit))
    print(f"PASS F21 paired source {tag}, {seen // 8} primary views per map and arm", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("held", choices=PEOPLE)
    cli.add_argument("seed", type=int, choices=(0, 1))
    cli.add_argument("--smoke", action="store_true")
    a = cli.parse_args()
    run(a.held, a.seed, a.smoke)
