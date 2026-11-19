"""Fit the matched four-arm 40-epoch hierarchy/jitter factorial."""
import argparse
from pathlib import Path

import numpy as np
import torch

from anchors import DATA_SHA, ORIGIN_SHA, PEOPLE, RUNTIME, row_sha, sha, split
from hierarchy_source import F17_ACTOR, HERE, KINDS, controls
from perturb import SHIFT, Feed
from produce_distill import publish
from source_targets import normalizer
from tree_head import install, train, verify_loss


F14 = HERE.parent / "friday_s1c_training_20261118"
F17 = HERE.parent / "friday_s1c_phase_20261118"


def run(held, seed, smoke):
    import braindecode
    import encoder312
    import model322
    from source_labels import read_training_labels

    if (sha(F17 / "actor.sha256") != F17_ACTOR or sha(F14 / "runtime.lock") != RUNTIME or
            sha(F14 / "original/source323.sha256") != ORIGIN_SHA or
            Path(model322.__file__).resolve() != F14 / "original/model322.py"):
        raise ValueError("exact F17 source recipe differs")
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name() or braindecode.__version__ != "1.8.1":
        raise ValueError("requires original source H100 runtime")
    torch.set_num_threads(2)
    path = F14 / "proteus_windows.npz"
    if sha(path) != DATA_SHA:
        raise ValueError("fixed PROTEUS source waveform differs")
    with np.load(path, allow_pickle=False) as z:
        x, subject, visit = (z[k] for k in ("X", "subject", "session"))
    if x.shape != (2655, 41, 480) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError("official waveform input differs")
    rows, stream, people = split(subject, visit, held)
    rng = torch.random.get_rng_state(), torch.cuda.get_rng_state()
    ref, batches, direction, masks, tape = controls(F17, F17, held, seed, smoke, rows, stream, people)
    if not torch.equal(rng[0], torch.random.get_rng_state()) or not torch.equal(rng[1], torch.cuda.get_rng_state()):
        raise ValueError("same F17 row tape consumed a model RNG")
    y = read_training_labels(path, rows, len(x))
    x, y, scale, mu, sd, weights, prepared = model322.setup(dict(X=x[rows], y=y, stream=stream), np.arange(len(rows)),
                                                             "cuda", epochs=ref["hard"]["epochs"])
    b = [r for ep in batches for r in ep]
    if any(not np.array_equal(a, c) for a, c in zip(b, (r for ep in prepared for r in ep), strict=True)):
        raise ValueError("same source batch schedule changed")
    if scale != ref["hard"]["scale"] or abs(sum(int(m.sum()) for m in masks) * 4 - sum(map(len, masks))):
        raise ValueError("source scale or exact F17 jitter dose differs")
    direction = torch.as_tensor(direction, device=x.device)
    jittered = torch.where(direction[:, None, None], x.roll(SHIFT, -1), x.roll(-SHIFT, -1))
    tag = f"{'smoke_' if smoke else ''}{seed}_{held}"
    baseline = ref["hard"]
    info = dict(held=held, seed=seed, epochs=baseline["epochs"], scale=scale, dataset=DATA_SHA, origin=ORIGIN_SHA,
                runtime=RUNTIME, actor=sha(HERE / "actor.sha256"), batches=baseline["batches"],
                source_rows=row_sha(rows), source_people=people, flat_initial=baseline["initial"], tape=tape,
                flat_origin=baseline["checkpoint"], jitter_origin=ref["jitter"]["checkpoint"], rng_isolated=True,
                train_rows_seen=baseline["train_rows_seen"], planned_jitter_rows=baseline["planned_rows"],
                held_labels_opened=False, device=torch.cuda.get_device_name(), torch=str(torch.__version__),
                braindecode=braindecode.__version__)
    states, initial_tree = set(), None
    for kind in KINDS:
        model = model322.Factor("session", False, "cuda")
        normalizer(model)
        if model.skip or encoder312.fingerprint(model) != baseline["initial"]:
            raise ValueError("unaligned exact flat S1C initialization")
        hierarchical = kind.startswith("hierarchy")
        use_jitter = kind.endswith("jitter")
        changed, proof, q_gradients = {}, {}, []
        if hierarchical:
            rng = torch.random.get_rng_state(), torch.cuda.get_rng_state()
            install(model)
            if not torch.equal(rng[0], torch.random.get_rng_state()) or not torch.equal(rng[1], torch.cuda.get_rng_state()):
                raise ValueError("hierarchy initialization used a new random draw")
            init_head = model.task.weight.detach().clone()
            init_body = encoder312.fingerprint(model.body)
            tree = encoder312.fingerprint(model)
            if initial_tree is not None and initial_tree != tree:
                raise ValueError("hierarchy/jitter pair did not share initialization")
            initial_tree = tree
            with torch.no_grad():
                model.moment_mean.copy_(mu)
                model.moment_sd.copy_(sd)
            probe = next(r for r in b if torch.unique(y[r]).numel() == 3)
            proof = verify_loss(model, x[probe], y[probe], weights)
        else:
            tree = baseline["initial"]
            hook = model.task.weight.register_hook(lambda grad: q_gradients.append(torch.count_nonzero(grad[:, -287:])))
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if ((not hierarchical and params != baseline["parameters"]) or abs(params / baseline["parameters"] - 1.) > .01):
            raise ValueError("factorial nominal trainable parameter count differs by more than 1%")
        native = model.forward
        feed = Feed(jittered, batches, masks) if use_jitter else None
        if feed is not None:
            model.forward = feed.install(model, native, x)
        if hierarchical:
            fit = train(model, x, y, weights, batches, seed)
            movement = (model.task.weight - init_head).detach().abs().amax(1)
            if not (movement > 0).all() or encoder312.fingerprint(model.body) == init_body:
                raise ValueError("both hierarchy heads and the raw body must update")
            changed = dict(motor_weight_delta=float(movement[0]), conditional_weight_delta=float(movement[1]),
                           body_updated=True)
        else:
            fit = model322.train(model, x, y, mu, sd, weights, batches)
            hook.remove()
            if len(q_gradients) != fit["steps"] or torch.stack(q_gradients).sum().item():
                raise ValueError("inactive engineered feature columns received gradients")
        model.forward = native
        state = encoder312.fingerprint(model)
        if (fit["steps"] != baseline["steps"] or feed is not None and feed.calls != baseline["steps"] or
                state in states or state == tree or not hierarchical and state != ref[kind]["state"]):
            raise ValueError(f"original fit, jitter tape or active learned state differs for {kind}")
        states.add(state)
        publish(HERE / f"{kind}_{tag}.pt", model, dict(info, kind=kind, state=state, initial=tree, parameters=params,
                head_rows=2 if hierarchical else 3, q_gradients_nonzero=0, engineered_features=0,
                jitter_rows=baseline["planned_rows"] if use_jitter else 0, **fit, **proof, **changed))
    print(f"PASS F19 paired source {tag}, exact flat and jitter controls, {baseline['steps']} steps per arm", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("held", choices=PEOPLE)
    cli.add_argument("seed", type=int, choices=(0, 1))
    cli.add_argument("--smoke", action="store_true")
    a = cli.parse_args()
    run(a.held, a.seed, a.smoke)
