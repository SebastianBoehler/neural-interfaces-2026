"""Fit the original raw-only waveform body with matched in-batch donor maps."""
import argparse
from pathlib import Path

import numpy as np
import torch

from anchors import DATA_SHA, ORIGIN_SHA, PEOPLE, RUNTIME, row_sha, sha, split
from donor_tape import CROSS, WITHIN, make_tape, verify_tape
from latent_fit import MASS, LatentMix, mechanism, probe
from mix_source import HERE, KINDS
from phase_source import F16_ACTOR, authenticated
from produce_distill import publish, write_bank
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
        raise ValueError("exact F16 source training actor has changed")
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name() or braindecode.__version__ != "1.8.1":
        raise ValueError("expected the original H100 runtime")
    torch.set_num_threads(2)
    path = F14 / "proteus_windows.npz"
    if sha(path) != DATA_SHA:
        raise ValueError("wrong official waveform input")
    with np.load(path, allow_pickle=False) as z:
        x, subject, visit = (z[k] for k in ("X", "subject", "session"))
    if x.shape != (2655, 41, 480) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError("official EEG shape or dtype differs")
    source, stream, people = split(subject, visit, held)
    ref, batches = authenticated(F16, F16, held, seed, smoke, source, stream, people)
    rng = torch.random.get_rng_state(), torch.cuda.get_rng_state()
    y = read_training_labels(path, source, len(x))
    step, slots = make_tape(batches, subject[source], stream, y, held, seed)
    bounds, real, guards = verify_tape(batches, subject[source], stream, step, slots, labels=y)
    tag = f"{'smoke_' if smoke else ''}{seed}_{held}"
    bank = write_bank(HERE / f"donor_{tag}.npz", index=source, batch=step, positions=slots)
    print(f"SEALED F20 source-only donor tape {bank}", flush=True)
    dat = dict(X=x[source], y=y, stream=stream)
    x, y, scale, mu, sd, weights, prepared = model322.setup(dat, np.arange(len(source)), "cuda", epochs=ref["epochs"])
    original = [b for ep in batches for b in ep]
    if scale != ref["scale"] or any(not np.array_equal(a, b) for a, b in zip(
            original, (r for ep in prepared for r in ep), strict=True)):
        raise ValueError("source waveform scale or native SessionBN schedule differs")
    if not torch.equal(rng[0], torch.random.get_rng_state()) or not torch.equal(rng[1], torch.cuda.get_rng_state()):
        raise ValueError("preparing the matched donor tape changed the model RNG")
    fixed = dict(held=held, seed=seed, epochs=ref["epochs"], scale=scale, dataset=DATA_SHA, origin=ORIGIN_SHA,
                runtime=RUNTIME, weights_origin=ref["checkpoint"], initial=ref["initial"], parameters=347903,
                batches=ref["batches"], source_rows=row_sha(source), source_people=people, donor_bank=bank,
                actor=sha(HERE / "actor.sha256"), rng_isolated=True, held_labels_opened=False, mass=MASS,
                device=torch.cuda.get_device_name(), torch=str(torch.__version__), braindecode=braindecode.__version__,
                **guards)
    states = set()
    for kind, order in zip(KINDS, (None, WITHIN, CROSS), strict=True):
        model = model322.Factor("session", False, "cuda")
        normalizer(model)
        initial = encoder312.fingerprint(model)
        if initial != ref["initial"] or model.skip or sum(p.numel() for p in model.parameters() if p.requires_grad) != 347903:
            raise ValueError("source model did not use the original zero-feature raw architecture")
        native = model.forward
        feed = LatentMix(bounds, slots, order, x.device) if order is not None else None
        evidence = {}
        if feed is not None:
            q, d = feed.plan[int(step[0])]
            r = original[int(step[0])]
            evidence = probe(model, x[r], y[r], q, d)
            model.forward = feed.install(model)
        skip_grads = []
        hook = model.task.weight.register_hook(lambda g: skip_grads.append(torch.count_nonzero(g[:, -287:])))
        fit = model322.train(model, x, y, mu, sd, weights, batches)
        hook.remove()
        model.forward = native
        state = encoder312.fingerprint(model)
        if (fit["steps"] != ref["steps"] or len(skip_grads) != ref["steps"] or torch.stack(skip_grads).sum().item() or
                state == initial or state in states or kind == "hard" and state != ref["state"] or
                feed is not None and (feed.calls != ref["steps"] or feed.mixed != guards["matched_rows"])):
            raise ValueError("inactive actual donor rows or exact F16 source hard control differs")
        diagnosis = mechanism(model, x, stream, real[:64])
        if state != encoder312.fingerprint(model):
            raise ValueError("source mechanism diagnostic altered fitted weights")
        states.add(state)
        publish(HERE / f"{kind}_{tag}.pt", model, dict(fixed, kind=kind, state=state, q_gradients_nonzero=0,
                mixed_rows=feed.mixed if feed is not None else 0, **fit, **evidence, **diagnosis))
    print(f"PASS F20 paired source {tag}, {guards['matched_rows']}/{guards['train_rows_seen']} mixed appearances", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("held", choices=PEOPLE)
    cli.add_argument("seed", type=int, choices=(0, 1))
    cli.add_argument("--smoke", action="store_true")
    a = cli.parse_args()
    run(a.held, a.seed, a.smoke)
