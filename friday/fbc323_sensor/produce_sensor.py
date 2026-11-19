"""Fit exact S1C hard CE and matched local or dispersed sensor withholding."""
import argparse
from pathlib import Path

import numpy as np
import torch

from anchors import DATA_SHA, ORIGIN_SHA, PEOPLE, RUNTIME, row_sha, sha, split
from perturb import Feed
from phase_source import F16_ACTOR, authenticated
from produce_distill import publish, write_bank
from withholding import MONTAGE, plan, zero_channels
from source_targets import normalizer, raw_logits


HERE = Path(__file__).resolve().parent
KINDS = ("hard", "local", "dispersed")
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
        raise ValueError("F16 same-platform training actor has changed")
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name() or braindecode.__version__ != "1.8.1":
        raise ValueError("expected the exact Ferranti H100 runtime")
    torch.set_num_threads(2)
    path = F14 / "proteus_windows.npz"
    if sha(path) != DATA_SHA:
        raise ValueError("official source tensor differs")
    with np.load(path, allow_pickle=False) as z:
        x, subject, visit, channels = (z[k] for k in ("X", "subject", "session", "ch_names"))
    if x.shape != (2655, 41, 480) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError("invalid official waveform tensor")
    rows, stream, people = split(subject, visit, held)
    ref, batches = authenticated(F16, F16, held, seed, smoke, rows, stream, people)
    cpu, cuda = torch.random.get_rng_state(), torch.cuda.get_rng_state()
    x = x[rows]
    a, b, masks, tape, guards = plan(x, channels, batches, held, seed)
    tag = f"{'smoke_' if smoke else ''}{seed}_{held}"
    bank = write_bank(HERE / f"mask_{tag}.npz", index=rows, nearby=a, dispersed=b)
    print(f"SEALED F18 label-blind sensor plan {bank}", flush=True)
    y = read_training_labels(path, rows, len(subject))
    data = dict(X=x, y=y, stream=stream)
    x, y, scale, mu, sd, weights, prepared = model322.setup(data, np.arange(len(rows)), "cuda", epochs=ref["epochs"])
    steps = [r for ep in batches for r in ep]
    if scale != ref["scale"] or any(not np.array_equal(a, b) for a, b in zip(
            steps, (r for ep in prepared for r in ep), strict=True)):
        raise ValueError("S1C initialization saw altered source scale or schedule")
    local, dispersed = (zero_channels(x, choices) for choices in (a, b))
    if not torch.equal(cpu, torch.random.get_rng_state()) or not torch.equal(cuda, torch.cuda.get_rng_state()):
        raise ValueError("label-blind spatial masking consumed a model RNG")
    replaced, total = sum(int(m.sum()) for m in masks), sum(map(len, masks))
    if replaced * 4 != total:
        raise ValueError("wrong source augmentation dose")
    info = dict(held=held, seed=seed, epochs=ref["epochs"], scale=scale, dataset=DATA_SHA, origin=ORIGIN_SHA,
                runtime=RUNTIME, weights_origin=ref["checkpoint"], initial=ref["initial"], parameters=347903,
                batches=ref["batches"], source_rows=row_sha(rows), source_people=people,
                actor=sha(HERE / "actor.sha256"), train_rows_seen=total, planned_rows=replaced, tape=tape, mask_bank=bank, montage=MONTAGE,
                held_labels_opened=False, rng_isolated=True, device=torch.cuda.get_device_name(),
                torch=str(torch.__version__), braindecode=braindecode.__version__, **guards)
    states = set()
    for kind, alternate in zip(KINDS, (None, local, dispersed), strict=True):
        model = model322.Factor("session", False, "cuda")
        normalizer(model)
        initial = encoder312.fingerprint(model)
        if initial != ref["initial"] or model.skip or sum(p.numel() for p in model.parameters() if p.requires_grad) != 347903:
            raise ValueError("unpaired original zero-feature S1C actor")
        feed = Feed(alternate, batches, masks) if alternate is not None else None
        native = model.forward
        if feed is not None:
            model.forward = feed.install(model, native, x)
        q_activity = []
        handle = model.task.weight.register_hook(lambda grad: q_activity.append(torch.count_nonzero(grad[:, -287:])))
        fit = model322.train(model, x, y, mu, sd, weights, batches)
        handle.remove()
        model.forward = native
        state = encoder312.fingerprint(model)
        if (fit["steps"] != ref["steps"] or len(q_activity) != ref["steps"] or torch.stack(q_activity).sum().item() or
                state == initial or state in states or kind == "hard" and state != ref["state"] or
                feed is not None and feed.calls != ref["steps"]):
            raise ValueError("original hard state, augmented feed, q-gradient or source work differs")
        if smoke:
            with torch.no_grad():
                sample = x[steps[0][:10]]
                if not torch.equal(model(sample), raw_logits(model, sample)):
                    raise ValueError("raw-only logits differ from native zero-feature forward")
        states.add(state)
        publish(HERE / f"{kind}_{tag}.pt", model, dict(info, kind=kind, state=state, q_gradients_nonzero=0,
                perturbed_rows=replaced if feed is not None else 0, **fit))
    print(f"PASS F18 paired source {tag}, {replaced}/{total} matched rows, max/mean energy error {guards['energy_error_max']:.3g}/{guards['energy_error_mean']:.3g}", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("held", choices=PEOPLE)
    cli.add_argument("seed", type=int, choices=(0, 1))
    cli.add_argument("--smoke", action="store_true")
    a = cli.parse_args()
    run(a.held, a.seed, a.smoke)
