"""Run each three-arm F16 fold after authenticating its F15 teacher."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from anchors import (DATA_SHA, F15_ACTOR, HERE, KINDS, ORIGIN_SHA, PEOPLE, RUNTIME, anchor, row_sha, schedule, sha,
                     split)
from source_targets import donor_plan, normalizer, raw_logits, teacher_bank
from student_fit import ALPHA, TEMPERATURE, soft_probe, train_soft


F14 = HERE.parent / "friday_s1c_training_20261118"
F15 = HERE.parent / "friday_s1c_skip_20261118"
F14_ACTOR = "74b2c7deeffd04d97a1355a437bc1e8704970d5f8e2ffbf426594ff0cf0b9824"


def write_bank(path, **fields):
    with path.open("xb") as handle:
        np.savez_compressed(handle, **fields)
    return sha(path)


def publish(path, model, receipt):
    payload = {k: receipt[k] for k in ("kind", "held", "seed", "scale")}
    with path.open("xb") as handle:
        torch.save(dict(payload, state=model.cpu().state_dict()), handle)
    with path.with_suffix(".json").open("x") as handle:
        json.dump(dict(receipt, checkpoint=sha(path)), handle, sort_keys=True, allow_nan=False)
        handle.write("\n")


def run(held, seed, smoke):
    import braindecode
    import encoder312
    import model322
    from source_labels import read_training_labels

    controls = {F14 / "actor.sha256": F14_ACTOR, F15 / "actor.sha256": F15_ACTOR,
                F14 / "runtime.lock": RUNTIME, F14 / "original/source323.sha256": ORIGIN_SHA}
    if (any(sha(p) != expected for p, expected in controls.items()) or
            Path(model322.__file__).resolve() != F14 / "original/model322.py"):
        raise ValueError("original F14 and F15 training source differs")
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name() or braindecode.__version__ != "1.8.1":
        raise ValueError("the exact Ferranti H100 runtime is required")
    torch.set_num_threads(2)
    path = F14 / "proteus_windows.npz"
    if sha(path) != DATA_SHA:
        raise ValueError("wrong official waveform input")
    with np.load(path, allow_pickle=False) as z:
        x, subject, visit = (z[k] for k in ("X", "subject", "session"))
    if x.shape != (2655, 41, 480) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError("wrong fixed online waveform shape")
    rows, stream, roster = split(subject, visit, held)
    teacher, t = anchor(F15, "dropout", held, seed, False, rows, stream, roster)
    hard, h = anchor(F15, "zero", held, seed, smoke, rows, stream, roster)
    if t["scale"] != h["scale"] or t["initial"] != h["initial"] or t["active_mask_rows"] != t["planned_mask_rows"]:
        raise ValueError("F15 teacher or hard baseline is not matched")
    x = x[rows]
    tag = f"{'smoke_' if smoke else ''}{seed}_{held}"
    logits, guards = teacher_bank(teacher, t, x, rows, stream)
    teacher_receipt = dict(checkpoint=t["checkpoint"], receipt=sha(teacher.with_suffix(".json")), guards=guards,
                           source_rows=row_sha(rows), teacher_labels_opened=False)
    bank_path = HERE / f"teacher_{tag}.npz"
    bank = write_bank(bank_path, index=rows, logits=logits, metadata=json.dumps(teacher_receipt, sort_keys=True))
    print(f"SEALED teacher labels unopened {bank}  {bank_path.name}", flush=True)
    y = read_training_labels(path, rows, len(subject))
    epochs = 4 if smoke else model322.EPOCHS
    batches, group_hash = schedule(rows, stream, seed, epochs)
    indexed, donors, counts = donor_plan(batches, rows, stream, y, held, seed)
    plan_path = HERE / f"plan_{tag}.npz"
    plan = write_bank(plan_path, index=indexed, donor=donors, groups=np.asarray(counts))
    print(f"SEALED matched target schedule {plan}  {plan_path.name}", flush=True)
    d = dict(X=x, y=y, stream=stream)
    x, y, scale, mu, sd, weights, fitted_batches = model322.setup(d, np.arange(len(rows)), "cuda", epochs=epochs)
    if scale != t["scale"] or any(not np.array_equal(a, b) for a, b in zip(
            (a for ep in batches for a in ep), (b for ep in fitted_batches for b in ep), strict=True)):
        raise ValueError("S1C setup changed the verified scale or source schedule")
    soft = torch.softmax(torch.from_numpy(logits).to(x.device) / TEMPERATURE, dim=1)
    target_trial = soft[np.searchsorted(rows, indexed)]
    target_donor = soft[np.searchsorted(rows, donors)]
    target_delta = float((target_trial - target_donor).abs().max())
    if not torch.isfinite(soft).all() or target_delta <= 1e-4:
        raise ValueError("matched deranging did not change trial-specific targets")
    info = dict(held=held, seed=seed, epochs=epochs, scale=scale, source_rows=row_sha(rows), source_people=roster,
                dataset=DATA_SHA, origin=ORIGIN_SHA, actor=sha(HERE / "actor.sha256"), runtime=RUNTIME,
                batches=group_hash, teacher=t["checkpoint"], teacher_bank=bank, donor_plan=plan,
                hard_origin=h["checkpoint"], hard_origin_receipt=sha(hard.with_suffix(".json")),
                max_target_delta=target_delta, source_only_labels=True, teacher_overlap="in_sample",
                held_labels_opened=False, device=torch.cuda.get_device_name(),
                torch=str(torch.__version__), braindecode=braindecode.__version__)
    learned = set()
    for kind, target in zip(KINDS, (None, target_trial, target_donor), strict=True):
        actor = model322.Factor("session", False, "cuda")
        normalizer(actor)
        if actor.skip or encoder312.fingerprint(actor) != h["initial"] or sum(
                p.numel() for p in actor.parameters() if p.requires_grad) != h["parameters"]:
            raise ValueError("student changed native F15 zero-q initialization or structure")
        q_grads = []

        def capture(grad):
            q_grads.append(torch.count_nonzero(grad[:, -287:]).detach())

        handle = actor.task.weight.register_hook(capture)
        probe = {}
        if kind == "hard":
            loss = model322.train(actor, x, y, mu, sd, weights, batches)
            with torch.no_grad():
                sample = x[batches[0][0]][:10]
                if not torch.equal(actor(sample), raw_logits(actor, sample)):
                    raise ValueError("raw-only head differs from the actual F15 zero-feature head")
        else:
            sample = batches[0][0]
            probe = soft_probe(actor, x, mu, sd, sample, target[0, :len(sample)])
            loss = train_soft(actor, x, y, mu, sd, weights, batches, target, seed)
        handle.remove()
        state = encoder312.fingerprint(actor)
        if (len(q_grads) != loss["steps"] or loss["steps"] != h["steps"] or
                torch.stack(q_grads).sum().item() != 0 or state in learned or state == h["initial"] or
                kind == "hard" and state != h["state"]):
            raise ValueError(f"bad q gradient, fit or exact F15 hard control for {kind}")
        learned.add(state)
        publish(HERE / f"{kind}_{tag}.pt", actor, dict(info, kind=kind, state=state, initial=h["initial"],
                parameters=h["parameters"], alpha=0. if target is None else ALPHA, temperature=TEMPERATURE,
                q_gradients_nonzero=0, **probe, **loss))
    print(f"PASS F16 {tag}, {len(rows)} source rows, {epochs} epochs, exact F15 hard CE", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("held", choices=PEOPLE)
    cli.add_argument("seed", choices=(0, 1), type=int)
    cli.add_argument("--smoke", action="store_true")
    args = cli.parse_args()
    run(args.held, args.seed, args.smoke)
