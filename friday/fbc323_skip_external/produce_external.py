"""Fit all-source F15 decoders on the frozen Ferranti H100 runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from fit import DATA_SHA, ORIGINAL_SOURCE, model_layer, row_sha, sha
from skip_policy import TrainingPolicy, mask_plan, verify_real_input


HERE = Path(__file__).resolve().parent
F14 = HERE.parent / "friday_s1c_training_20261118"
F15 = HERE.parent / "friday_s1c_skip_20261118"
F14_ACTOR = "74b2c7deeffd04d97a1355a437bc1e8704970d5f8e2ffbf426594ff0cf0b9824"
F15_ACTOR = "04ec8dd8e7593088a1b882e8fcf75cf3ac9bf67ff737c14c1903b70f97ea1be3"
RUNTIME = "da21360c9a85d7561eee8f0f68367b545797641a69e3726354275cbf404ec30e"
KINDS = ("original", "scaled", "dropout")


def source_only():
    from source_labels import read_training_labels

    path = F14 / "proteus_windows.npz"
    if sha(path) != DATA_SHA:
        raise ValueError("PROTEUS training tensor differs")
    with np.load(path, allow_pickle=False) as z:
        x, subject, visit = (z[k] for k in ("X", "subject", "session"))
    people = sorted(set(subject))
    ids = tuple(s.rsplit("/", 1)[-1] for s in people)
    if (ids != tuple(f"P{i:02d}" for i in range(1, 11)) or
            x.shape != (2655, 41, 480) or x.dtype != np.float32 or not np.isfinite(x).all()):
        raise ValueError("wrong all-PROTEUS source roster or raw inputs")
    rows = np.arange(len(x))
    return dict(X=x, y=read_training_labels(path, rows, len(x)),
                stream=np.char.add(np.char.add(subject, "|"), visit)), rows, people


def publish(model, record, smoke):
    tag = f"{'smoke_' if smoke else ''}{record['kind']}_{record['seed']}_all_source"
    path = HERE / f"{tag}.pt"
    tmp = path.with_suffix(".tmp")
    payload = {k: record[k] for k in ("kind", "held", "seed", "scale")}
    with tmp.open("xb") as handle:
        torch.save(dict(payload, state=model.cpu().state_dict()), handle)
    os.link(tmp, path)
    tmp.unlink()
    with path.with_suffix(".json").open("x") as handle:
        json.dump(dict(record, checkpoint=sha(path)), handle, sort_keys=True, allow_nan=False)
        handle.write("\n")


def run(seed, smoke):
    import braindecode
    import encoder312
    import fit
    import model322
    import skip_policy
    from session_bn import SessionBN

    required = {F14 / "actor.sha256": F14_ACTOR, F15 / "actor.sha256": F15_ACTOR, F14 / "runtime.lock": RUNTIME}
    if (any(sha(p) != digest for p, digest in required.items()) or
            Path(fit.__file__).resolve() != F14 / "fit.py" or
            Path(model322.__file__).resolve() != F14 / "original/model322.py" or
            Path(skip_policy.__file__).resolve() != F15 / "skip_policy.py"):
        raise ValueError("original fitted recipe source must match F14 and F15")
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name() or braindecode.__version__ != "1.8.1":
        raise ValueError("requires exact original same-platform Ferranti H100 runtime")
    torch.set_num_threads(2)
    data, rows, people = source_only()
    model322.SEED = seed
    epochs = 4 if smoke else model322.EPOCHS
    x, y, scale, mu, sd, weights, batches = model322.setup(data, rows, "cuda", epochs=epochs)
    schedule = hashlib.sha256(b"".join(rows[b].astype("<i8").tobytes() for epoch in batches for b in epoch)).hexdigest()
    masks, plan = mask_plan(batches, seed, "P00")
    planned, total = sum(int(m.sum()) for m in masks), sum(len(m) for m in masks)
    if 4 * planned != total:
        raise ValueError("source mask rate must equal one quarter")
    masks = [m.to(x.device) for m in masks]
    info = dict(held="all_source", seed=seed, epochs=epochs, source_rows=row_sha(rows), source_people=people,
                batches=schedule, scale=scale, origin=ORIGINAL_SOURCE, dataset=DATA_SHA, mask_plan=plan,
                planned_mask_rows=planned, train_rows_seen=total, actor=sha(HERE / "actor.sha256"),
                runtime=RUNTIME, device=torch.cuda.get_device_name(), torch=str(torch.__version__),
                braindecode=braindecode.__version__, held_labels_opened=False, mask_rng_isolated=True)
    reference_initial, observed_states = None, []
    for kind in KINDS:
        model = model322.Factor("session", True, "cuda")
        layer = model_layer(model, SessionBN)
        initial = encoder312.fingerprint(model)
        parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if reference_initial is None:
            reference_initial = initial
        if initial != reference_initial or parameters != 347903 or not model.skip or layer.causal:
            raise ValueError("paired initialization, parameter count or original training path differs")
        if smoke and kind == "original":
            verify_real_input(model, x[batches[0][0]], masks[0], mu, sd)
            if encoder312.fingerprint(model) != initial:
                raise ValueError("the verification changed the fitted actor")
        policy = TrainingPolicy(kind, masks) if kind != "original" else None
        native = model.forward
        if policy is not None:
            model.forward = policy.install(model)
        q_gradients = []
        handle = model.task.weight.register_hook(lambda grad: q_gradients.append(grad[:, -287:].detach().abs().max()))
        result = model322.train(model, x, y, mu, sd, weights, batches)
        handle.remove()
        model.forward = native
        grad = torch.stack(q_gradients)
        if (len(grad) != len(masks) or result["steps"] != len(masks) or
                policy is not None and policy.calls != len(masks) or
                not torch.isfinite(grad).all() or not (grad > 0).all()):
            raise ValueError(f"invalid source fixed-feature gradient or optimizer work for {kind}")
        state = encoder312.fingerprint(model)
        if state == initial or state in observed_states:
            raise ValueError("untrained or duplicated checkpoint")
        observed_states.append(state)
        publish(model, dict(info, kind=kind, initial=initial, state=state, parameters=parameters,
                            active_mask_rows=planned if kind == "dropout" else 0,
                            max_q_gradient=float(grad.max()), q_gradient_active_steps=len(grad), **result), smoke)
        del model, layer
    print(f"PASS F15e paired all-source seed {seed}, {epochs} epochs, {planned}/{total} masked rows", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("seed", choices=(0, 1), type=int)
    cli.add_argument("--smoke", action="store_true")
    args = cli.parse_args()
    run(args.seed, args.smoke)
