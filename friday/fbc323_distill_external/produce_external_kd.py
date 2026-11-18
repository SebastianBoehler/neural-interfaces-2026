"""Train F16 raw-only students on PROTEUS, without mounting Scherer data."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from anchors import DATA_SHA, KINDS, ORIGIN_SHA, RUNTIME, row_sha, schedule, sha
from external_source import F16_ACTOR, HERE, all_source, teacher_receipt
from produce_distill import publish, write_bank
from source_targets import donor_plan, normalizer, raw_logits, teacher_bank
from student_fit import ALPHA, TEMPERATURE, soft_probe, train_soft


F14 = HERE.parent / "friday_s1c_training_20261118"
F15E = HERE.parent / "friday_s1c_skip_external_20261118"
F16 = HERE.parent / "friday_s1c_distill_20261118"


def train(seed, smoke):
    import braindecode
    import encoder312
    import model322
    from source_labels import read_training_labels

    if (sha(F16 / "actor.sha256") != F16_ACTOR or sha(F14 / "runtime.lock") != RUNTIME or
            sha(F14 / "original/source323.sha256") != ORIGIN_SHA or
            Path(model322.__file__).resolve() != F14 / "original/model322.py"):
        raise ValueError("original matched training implementation differs")
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name() or braindecode.__version__ != "1.8.1":
        raise ValueError("requires exact Ferranti H100 F16 runtime")
    torch.set_num_threads(2)
    path = F14 / "proteus_windows.npz"
    x, rows, stream, people = all_source(path, waveforms=True)
    teacher, t = teacher_receipt(F15E, F15E, rows, stream, people, seed)
    tag = f"{'smoke_' if smoke else ''}{seed}_all_source"
    logits, guards = teacher_bank(teacher, t, x, rows, stream)
    teacher_record = dict(checkpoint=t["checkpoint"], receipt=sha(teacher.with_suffix(".json")), guards=guards,
                          source_rows=row_sha(rows), teacher_labels_opened=False)
    bank = write_bank(HERE / f"teacher_{tag}.npz", index=rows, logits=logits,
                      metadata=json.dumps(teacher_record, sort_keys=True))
    print(f"SEALED source teacher before labels {bank}", flush=True)
    y = read_training_labels(path, rows, len(rows))
    epochs = 4 if smoke else model322.EPOCHS
    batches, trace = schedule(rows, stream, seed, epochs)
    mapped, donors, counts = donor_plan(batches, rows, stream, y, "P00", seed)
    plan = write_bank(HERE / f"plan_{tag}.npz", index=mapped, donor=donors, groups=np.asarray(counts))
    print(f"SEALED matched source target exposure {plan}", flush=True)
    x, y, scale, mu, sd, weights, fitted = model322.setup(dict(X=x, y=y, stream=stream), rows, "cuda", epochs=epochs)
    if scale != t["scale"] or any(not np.array_equal(a, b) for a, b in zip(
            (mm for ep in batches for mm in ep), (mm for ep in fitted for mm in ep), strict=True)):
        raise ValueError("source recipe differs from original F15e fit")
    soft = torch.softmax(torch.from_numpy(logits).to(x.device) / TEMPERATURE, dim=1)
    actual, control = soft[mapped], soft[donors]
    target_delta = float((actual - control).abs().max())
    if not torch.isfinite(soft).all() or target_delta <= 1e-4:
        raise ValueError("derangement does not vary the active soft supervision")
    fixed = dict(held="all_source", seed=seed, epochs=epochs, scale=scale, source_rows=row_sha(rows), source_people=people,
                 dataset=DATA_SHA, origin=ORIGIN_SHA, actor=sha(HERE / "actor.sha256"), runtime=RUNTIME, batches=trace,
                 teacher=t["checkpoint"], teacher_bank=bank, donor_plan=plan, initial=t["initial"], parameters=347903,
                 max_target_delta=target_delta, teacher_overlap="in_sample", held_labels_opened=False,
                 source_only_labels=True, device=torch.cuda.get_device_name(), torch=str(torch.__version__),
                 braindecode=braindecode.__version__, temperature=TEMPERATURE)
    states = set()
    for kind, target in zip(KINDS, (None, actual, control), strict=True):
        model = model322.Factor("session", False, "cuda")
        normalizer(model)
        if model.skip or encoder312.fingerprint(model) != t["initial"] or sum(
                p.numel() for p in model.parameters() if p.requires_grad) != 347903:
            raise ValueError("unpaired zero-q architecture or initialization")
        q_activity = []
        handle = model.task.weight.register_hook(lambda grad: q_activity.append(torch.count_nonzero(grad[:, -287:])))
        probe = {}
        sample = batches[0][0]
        if target is None:
            fit = model322.train(model, x, y, mu, sd, weights, batches)
            with torch.no_grad():
                voltage = x[sample[:10]]
                if not torch.equal(model(voltage), raw_logits(model, voltage)):
                    raise ValueError("logical zero feature output differs from actual no-feature head")
        else:
            probe = soft_probe(model, x, mu, sd, sample, target[0, :len(sample)])
            fit = train_soft(model, x, y, mu, sd, weights, batches, target, seed)
        handle.remove()
        state = encoder312.fingerprint(model)
        if (fit["steps"] != sum(map(len, batches)) or len(q_activity) != fit["steps"] or
                torch.stack(q_activity).sum().item() or state == t["initial"] or state in states):
            raise ValueError(f"incorrect zero-q gradients, source work or duplicate state for {kind}")
        states.add(state)
        publish(HERE / f"{kind}_{tag}.pt", model, dict(fixed, kind=kind, state=state, q_gradients_nonzero=0,
                                                       alpha=0. if target is None else ALPHA, **probe, **fit))
    print(f"PASS F16e paired source seed {seed}, {epochs} epochs, {mapped.size} matched row exposures", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("seed", type=int, choices=(0, 1))
    cli.add_argument("--smoke", action="store_true")
    args = cli.parse_args()
    train(args.seed, args.smoke)
