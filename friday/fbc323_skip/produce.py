"""Train the original F14 S1C recipe with four matched fixed-feature policies."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from fit import DATA_SHA, ORIGINAL_SOURCE, model_layer, row_sha, sha, source_only
from skip_policy import KINDS, TrainingPolicy, mask_plan, verify_real_input


HERE = Path(__file__).resolve().parent
F14 = HERE.parent / "friday_s1c_training_20261118"
F14_ACTOR = "74b2c7deeffd04d97a1355a437bc1e8704970d5f8e2ffbf426594ff0cf0b9824"
RUNTIME = "da21360c9a85d7561eee8f0f68367b545797641a69e3726354275cbf404ec30e"


def original_anchor(held, seed, smoke, info):
    if sha(F14 / "actor.sha256") != F14_ACTOR or sha(F14 / "runtime.lock") != RUNTIME:
        raise ValueError("same-platform F14 training release differs")
    tag = f"{'smoke_' if smoke else ''}original_{seed}_{held}"
    path = F14 / f"{tag}.json"
    meta = json.loads(path.read_text())
    controls = ("held", "seed", "epochs", "source_rows", "source_people", "batches", "scale", "dataset", "origin", "runtime")
    if any(meta[k] != info[k] for k in controls) or meta["actor"] != F14_ACTOR or meta["kind"] != "original":
        raise ValueError("source controls differ from the authentic F14 old arm")
    if sha(path.with_suffix(".pt")) != meta["checkpoint"] or meta["held_labels_opened"]:
        raise ValueError("authentic F14 old-arm checkpoint differs")
    return meta


def publish(model, record, smoke):
    tag = f"{'smoke_' if smoke else ''}{record['kind']}_{record['seed']}_{record['held']}"
    path = HERE / f"{tag}.pt"
    tmp = path.with_suffix(".tmp")
    payload = {k: record[k] for k in ("kind", "held", "seed", "scale")}
    with tmp.open("xb") as handle:
        torch.save(dict(payload, state=model.cpu().state_dict()), handle)
    os.link(tmp, path)
    tmp.unlink()
    receipt = path.with_suffix(".json")
    pending = path.with_suffix(".writing")
    with pending.open("x") as handle:
        json.dump(dict(record, checkpoint=sha(path)), handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.link(pending, receipt)
    pending.unlink()


def run(held, seed, smoke):
    import braindecode
    import encoder312
    import fit
    import model322
    from session_bn import SessionBN

    if Path(fit.__file__).resolve() != F14 / "fit.py" or Path(model322.__file__).resolve() != F14 / "original/model322.py":
        raise ValueError("requires the authentic F14 source loader and exact S1C model")
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name() or braindecode.__version__ != "1.8.1":
        raise ValueError("requires the exact same-platform Ferranti H100 training allocation")
    torch.set_num_threads(2)
    people = tuple(f"P{i:02d}" for i in range(1, 11))
    d, rows, roster = source_only(F14 / "proteus_windows.npz", held, people)
    model322.SEED = seed
    epochs = 4 if smoke else model322.EPOCHS
    x, y, scale, mu, sd, weights, batches = model322.setup(d, np.arange(len(rows)), "cuda", epochs=epochs)
    schedule = hashlib.sha256(b"".join(rows[b].astype("<i8").tobytes() for epoch in batches for b in epoch)).hexdigest()
    masks, plan = mask_plan(batches, seed, held)
    planned, total = sum(int(m.sum()) for m in masks), sum(len(m) for m in masks)
    if planned * 4 != total:
        raise ValueError("source mask dose does not match the deterministic control")
    masks = [m.to(x.device) for m in masks]
    info = dict(held=held, seed=seed, epochs=epochs, source_rows=row_sha(rows), source_people=roster,
                batches=schedule, scale=scale, origin=ORIGINAL_SOURCE, dataset=DATA_SHA, mask_plan=plan,
                planned_mask_rows=planned, train_rows_seen=total, actor=sha(HERE / "actor.sha256"),
                runtime=sha(F14 / "runtime.lock"), device=torch.cuda.get_device_name(), torch=str(torch.__version__),
                braindecode=braindecode.__version__, held_labels_opened=False, mask_rng_isolated=True)
    anchor = original_anchor(held, seed, smoke, info)
    info["f14_original_state"] = anchor["state"]
    observed_states = []
    for kind in KINDS:
        model = model322.Factor("session", kind != "zero", "cuda")
        layer = model_layer(model, SessionBN)
        initial = encoder312.fingerprint(model)
        parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if initial != anchor["initial"] or parameters != anchor["parameters"] or layer.causal:
            raise ValueError("paired initialization, parameter count or original training path differs")
        if smoke and kind == "original":
            verify_real_input(model, x[batches[0][0]], masks[0], mu, sd)
            if encoder312.fingerprint(model) != initial:
                raise ValueError("source probe touched the original training actor")
        policy = TrainingPolicy(kind, masks) if kind in ("scaled", "dropout") else None
        ordinary = model.forward
        if policy is not None:
            model.forward = policy.install(model)
        q_gradients = []
        handle = model.task.weight.register_hook(lambda grad: q_gradients.append(grad[:, -287:].detach().abs().max()))
        result = model322.train(model, x, y, mu, sd, weights, batches)
        handle.remove()
        model.forward = ordinary
        grad = torch.stack(q_gradients)
        if (len(grad) != len(masks) or result["steps"] != anchor["steps"] or
                policy is not None and policy.calls != result["steps"] or not torch.isfinite(grad).all()):
            raise ValueError("source fixed-branch gradient evidence misses a scheduled step")
        max_gradient, active_steps = float(grad.max()), int(torch.count_nonzero(grad))
        if (max_gradient == 0) != (kind == "zero"):
            raise ValueError(f"fixed-branch gradients differ from the original q path for {kind}")
        state = encoder312.fingerprint(model)
        if state == initial or state in observed_states:
            raise ValueError("missing or duplicated fit state")
        if kind == "original" and state != anchor["state"]:
            raise ValueError(f"the q-on source control differs from F14: {state} versus {anchor['state']}")
        observed_states.append(state)
        publish(model, dict(info, kind=kind, initial=initial, state=state, parameters=parameters,
                            active_mask_rows=planned if kind == "dropout" else 0,
                            max_q_gradient=max_gradient, q_gradient_active_steps=active_steps, **result), smoke)
        del model, layer
    print(f"PASS F15 paired {held} seed {seed}, {epochs} epochs, q-on F14 state exact, {planned}/{total} masked rows",
          flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("held", choices=tuple(f"P{i:02d}" for i in range(1, 11)))
    p.add_argument("seed", choices=(0, 1), type=int)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    run(a.held, a.seed, a.smoke)
