"""Fit matched source-only variants of the original §323 S1C model on Ferranti."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from bounded_train import bounded_training, verify_training


HERE = Path(__file__).resolve().parent
ORIGINAL_SOURCE = "88437e69b5eaf7ff84a3635df88fe68b38394abbc527629b0f41c7fcc24eb6d5"
DATA_SHA = "2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327"
KINDS = ("original", "causal", "bounded")


def sha(p):
    with p.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def row_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype="<i8").tobytes()).hexdigest()


def model_layer(model, cls):
    layers = [m for m in model.body.modules() if type(m) is cls]
    if len(layers) != 1 or (layers[0].m, layers[0].k, layers[0].k_train) != (32, 20., 20.):
        raise ValueError("actual S1C SessionBN layer differs")
    return layers[0]


def source_only(path, held, people):
    from source_labels import read_training_labels

    if sha(path) != DATA_SHA:
        raise ValueError("official source input identity differs")
    with np.load(path, allow_pickle=False) as z:
        x, subject, session = z["X"], z["subject"].astype(str), z["session"].astype(str)
    if x.shape != (2655, 41, 480) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError("official raw tensor differs")
    ids = np.array([s.rsplit("/", 1)[-1] for s in subject])
    fit = np.flatnonzero(ids != held)
    if set(ids[fit]) != set(people) - {held} or not (ids == held).any():
        raise ValueError("invalid original held-person fold")
    data = dict(X=x[fit], y=read_training_labels(path, fit, len(x)),
                stream=np.char.add(np.char.add(subject[fit], "|"), session[fit]))
    return data, fit, sorted(set(subject[fit]))


def fit(held, seed, smoke):
    import braindecode
    import encoder312
    import model322
    import session_bn
    import source_labels

    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name() or braindecode.__version__ != "1.8.1":
        raise ValueError("registered Ferranti H100 and original Braindecode required")
    origin = HERE / "original"
    for mod, source in ((model322, "model322.py"), (encoder312, "encoder312.py"),
                        (session_bn, "experiments/session_bn.py"), (source_labels, "source_labels.py")):
        if Path(mod.__file__).resolve() != origin / source:
            raise ValueError(f"source import differs: {mod.__name__}")
    if sha(origin / "source323.sha256") != ORIGINAL_SOURCE:
        raise ValueError("original S1C release identity differs")
    torch.set_num_threads(2)
    people = tuple(f"P{i:02d}" for i in range(1, 11))
    d, source_rows, fit_people = source_only(HERE / "proteus_windows.npz", held, people)
    model322.SEED = seed
    epochs = 4 if smoke else model322.EPOCHS
    x, y, scale, mu, sd, weights, batches = model322.setup(d, np.arange(len(source_rows)), "cuda", epochs=epochs)
    batch_rows = hashlib.sha256(b"".join(source_rows[b].astype("<i8").tobytes() for epoch in batches for b in epoch)).hexdigest()
    info = dict(held=held, seed=seed, epochs=epochs, source_rows=row_sha(source_rows), source_people=fit_people,
                batches=batch_rows, scale=scale, origin=ORIGINAL_SOURCE, dataset=DATA_SHA,
                actor=sha(HERE / "actor.sha256"), runtime=sha(HERE / "runtime.lock"),
                device=torch.cuda.get_device_name(), torch=str(torch.__version__), braindecode=braindecode.__version__,
                held_labels_opened=False)
    paired = []
    reference_initial = None
    for kind in KINDS:
        model = model322.Factor("session", True, "cuda")
        layer = model_layer(model, session_bn.SessionBN)
        initial = encoder312.fingerprint(model)
        parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if reference_initial is None:
            reference_initial = initial, parameters
        if (initial, parameters) != reference_initial or not model.skip or layer.causal:
            raise ValueError("paired training initialization, count or original S1C differs")
        if kind == "original" and smoke:
            state = encoder312.fingerprint(model)
            probe = copy.deepcopy(model)
            probe_layer = model_layer(probe, session_bn.SessionBN)
            captured = []
            handle = probe_layer.register_forward_pre_hook(lambda _, args: captured.append(args[0].detach().clone()))
            with torch.no_grad():
                probe(x[batches[0][0]])
            handle.remove()
            if encoder312.fingerprint(model) != state or len(captured) != 1:
                raise ValueError("cannot isolate actual source activation probe")
            verify_training(probe_layer, captured[0])
        layer.causal = kind != "original"
        activity = []
        native = layer.forward
        if kind == "bounded":
            layer.forward = MethodType(lambda self, xx: bounded_training(self, xx, activity), layer)
        details = model322.train(model, x, y, mu, sd, weights, batches)
        if kind == "bounded":
            layer.forward = native
        if details["steps"] != sum(map(len, batches)) or not all(np.isfinite(v) for v in details.values()):
            raise ValueError("original optimizer work or finite gradient checks failed")
        active = int(torch.stack(activity).sum()) if activity else 0
        if (kind == "bounded") != (active > 0) or kind == "bounded" and len(activity) != details["steps"]:
            raise ValueError(f"bounded source activity mismatch for {kind}: {active} clips, "
                             f"{len(activity)} calls, {details['steps']} optimizer steps")
        state = encoder312.fingerprint(model)
        if state == initial:
            raise ValueError("source checkpoint has no trained state")
        tag = f"{'smoke_' if smoke else ''}{kind}_{seed}_{held}"
        path = HERE / f"{tag}.pt"
        tmp = path.with_suffix(".tmp")
        with tmp.open("xb") as handle:
            torch.save(dict(state=model.cpu().state_dict(), scale=scale, kind=kind, seed=seed, held=held), handle)
        os.link(tmp, path)
        tmp.unlink()
        receipt = dict(info, kind=kind, initial=initial, parameters=parameters, state=state, checkpoint=sha(path),
                       active_training_clips=active, **details)
        rpt = path.with_suffix(".json")
        with rpt.open("x") as handle:
            json.dump(receipt, handle, sort_keys=True, allow_nan=False)
            handle.write("\n")
        paired.append(receipt)
        del model, layer
    if len({m["state"] for m in paired}) != len(KINDS):
        raise ValueError("different training objectives produced the same fitted state")
    print(f"PASS F14 {held} seed {seed} paired source fits, {epochs} epochs, {paired[-1]['active_training_clips']} clips",
          flush=True)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("held", choices=tuple(f"P{i:02d}" for i in range(1, 11)))
    args.add_argument("seed", choices=(0, 1), type=int)
    args.add_argument("--smoke", action="store_true")
    cli = args.parse_args()
    fit(cli.held, cli.seed, cli.smoke)
