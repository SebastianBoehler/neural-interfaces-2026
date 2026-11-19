"""Fit paired F37 controls on authentic all-PROTEUS source data only."""

import argparse
from pathlib import Path

import braindecode
import numpy as np
import torch

from anchors import DATA_SHA, ORIGIN_SHA, RUNTIME, row_sha, sha
from encoder312 import fingerprint
import model322
from perturb import Feed, random_tape
from produce_distill import publish
from source_labels import read_training_labels
from source_scope import authenticated, scope
from source_targets import normalizer, raw_logits
from spectral_shift import perturb


HERE = Path(__file__).resolve().parent
F14 = HERE.parent / "friday_s1c_training_20261118"
F16E = HERE.parent / "friday_s1c_distill_external_20261118"
ARMS = ("hard", "beta", "alpha")


def run(seed, smoke):
    path = F14 / "proteus_windows.npz"
    device = torch.cuda.get_device_name()
    if ("H100" not in device or braindecode.__version__ != "1.8.1" or
            sha(path) != DATA_SHA or sha(F14 / "runtime.lock") != RUNTIME or
            sha(F14 / "original/source323.sha256") != ORIGIN_SHA or
            Path(model322.__file__).resolve() != F14 / "original/model322.py"):
        raise ValueError("original all-source H100 renderer or data differ")
    torch.set_num_threads(2)
    eeg, rows, stream, people, _ = scope(path)
    ref, batches = authenticated(F16E, F16E, seed, smoke, rows, stream, people)
    labels = read_training_labels(path, rows, len(eeg))
    x, y, scale, mu, sd, weights, native = model322.setup(
        dict(X=eeg[rows], y=labels, stream=stream), np.arange(len(rows)), "cuda", epochs=ref["epochs"])
    flat = [r for ep in batches for r in ep]
    if scale != ref["scale"] or any(not np.array_equal(p, q) for p, q in
                                    zip(flat, (r for ep in native for r in ep), strict=True)):
        raise ValueError("original all-source scale or update schedule changed")
    cpu, gpu = torch.random.get_rng_state(), torch.cuda.get_rng_state()
    signs, directions, masks, tape = random_tape(batches, len(rows), "P00", seed)
    sampled = np.unique(np.concatenate([r[m] for r, m in zip(flat[:3], masks[:3], strict=True)]))[:8]
    beta, alpha, proof = perturb(x, directions, signs, sampled)
    treated, total = sum(int(m.sum()) for m in masks), sum(map(len, masks))
    if (not torch.equal(cpu, torch.random.get_rng_state()) or
            not torch.equal(gpu, torch.cuda.get_rng_state()) or treated * 4 != total):
        raise ValueError("the transform changed source dose or RNG")
    fixed = dict(held="all_source", seed=seed, epochs=ref["epochs"], scale=scale, dataset=DATA_SHA,
                 origin=ORIGIN_SHA, runtime=RUNTIME, source_rows=row_sha(rows), source_people=people,
                 batches=ref["batches"], source_count=len(rows), source_sessions=len(np.unique(stream)),
                 actor=sha(HERE / "actor.sha256"), weights_origin=ref["checkpoint"], initial=ref["initial"],
                 device=device, torch=str(torch.__version__), braindecode=braindecode.__version__,
                 held_labels_opened=False, scherer_exposed=False, source_labels_only=True,
                 parameters=347903, q_gradients_nonzero=0, tape=tape, treated=treated, total=total, **proof)
    states = {ref["initial"]}
    for arm, transformed in zip(ARMS, (None, beta, alpha), strict=True):
        model = model322.Factor("session", False, "cuda")
        normalizer(model)
        if fingerprint(model) != ref["initial"] or sum(p.numel() for p in model.parameters() if p.requires_grad) != 347903:
            raise ValueError("original raw initialization differs")
        original = model.forward
        feed = Feed(transformed, batches, masks) if transformed is not None else None
        if feed is not None:
            model.forward = feed.install(model, original, x)
        q = []
        hook = model.task.weight.register_hook(lambda gradient: q.append(gradient[:, -287:].count_nonzero()))
        try:
            fit = model322.train(model, x, y, mu, sd, weights, batches)
        finally:
            hook.remove()
            model.forward = original
        state = fingerprint(model)
        if (fit["steps"] != ref["steps"] or len(q) != ref["steps"] or torch.stack(q).sum().item() or
                state in states or feed is not None and feed.calls != ref["steps"] or
                not np.isfinite([fit["first_loss"], fit["last_loss"], fit["max_gradient"]]).all() or
                fit["max_gradient"] <= 0):
            raise ValueError("source schedule changed or treatment did not train independently")
        with torch.no_grad():
            sample = x[flat[0][:10]]
            if not torch.equal(model(sample), raw_logits(model, sample)):
                raise ValueError("native raw zero-feature equivalence failed")
        states.add(state)
        tag = f"{'smoke_' if smoke else ''}{seed}_all_source"
        target = HERE / f"{arm}_{tag}.pt"
        publish(target, model, dict(fixed, kind=arm, state=state, perturbed_rows=treated if feed else 0, **fit))
        if arm == "hard" and (state != ref["state"] or sha(target) != ref["checkpoint"]):
            raise ValueError("exact all-source hard bytes changed")
    print(f"PASS F37e native source {seed} {smoke} {treated}/{total}", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("seed", type=int, choices=(0, 1))
    cli.add_argument("--smoke", action="store_true")
    args = cli.parse_args()
    run(args.seed, args.smoke)
