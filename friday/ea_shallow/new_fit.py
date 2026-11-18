"""Train new source-excluded Shallow EA fits and seal causal predictions."""
import argparse
from copy import deepcopy
import math
import os
from pathlib import Path

import numpy as np

from probe import ORDERS, TOL, deliveries, digest, fingerprint, first_window, read_inputs, tensor_identity, write_json

HERE = Path(__file__).resolve().parent
REV = "c625ecc736a881836a25f85a741ea9147bf698bf"
PARENT = "a01929b52e7d7ba3ecaa001785c2f3a0811fd6dbb0564fde5070d0418af35713"
SOURCE = Path("/mnt/lustre/work/gehler/gfh527/ni26_friday_ea_matrix_20261118")
MODEL = "shallow_ea"
SEEDS = tuple(range(6))
PEOPLE = tuple(f"P{i:02d}" for i in range(1, 11))
LAYOUT = ("recording", *ORDERS)
EPOCHS, BATCH = 40, 64


def configure_runtime(torch):
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name():
        raise ValueError("Galvani A100 is required")
    if (torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), np.__version__) != (
        "2.14.1+cu126", "12.6", 91002, "2.5.4"
    ) or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("pinned deterministic runtime differs")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def provenance():
    return dict(cfg=MODEL, historical_reconstruction=False, shared_by_comparators=True,
                held_labels_opened=False, source=REV, parent_source_sha=PARENT,
                code_sha=digest(HERE / "source.sha256"), inputs_sha=digest(HERE / "inputs.sha256"))


def infer(actor, rows, torch):
    # The alignment is a prefix sum. The model itself receives one current window.
    aligned = actor._aligned(rows, True)
    x = torch.from_numpy(aligned).to("cuda")
    with torch.inference_mode():
        probs = torch.cat([actor.net(window) for window in x.split(1)]).cpu().double().softmax(1)
    return probs.numpy()


def replay(actor, features, test, source, delivery, torch):
    stream = actor.d["stream"]
    sessions = tuple(dict.fromkeys(stream[test]))
    out = np.full((len(test), 3), np.nan)
    checks = {}
    for session in sessions:
        recording = test[stream[test] == session]
        rows = delivery[stream[delivery] == session]
        if not np.array_equal(np.sort(rows), recording) or len(rows) < 16:
            raise ValueError("session delivery is incomplete")
        check = first_window(actor, features, recording, rows)
        prefix = rows[:16]
        full = infer(actor, rows, torch)
        before = infer(actor, prefix, torch)
        other = next((s for s in sessions if s != session), stream[source[0]])
        context = delivery if other in sessions else source
        infer(actor, context[stream[context] == other][:16], torch)
        after = infer(actor, prefix, torch)
        batch = actor.predict_proba_stream(prefix)
        batch_delta = float(np.abs(before - batch).max())
        if (check["first_direct_max_abs"] != 0 or not np.array_equal(full[:16], before)
                or not np.array_equal(before, after) or batch_delta > TOL
                or not np.array_equal(before.argmax(1), batch.argmax(1))):
            raise ValueError("first input, prefix, interleaved reset or historical actor equivalence failed")
        out[np.searchsorted(test, rows)] = full
        checks[session] = dict(**check, prefix_exact=True, reset_exact=True, batch_max_abs=batch_delta)
    if not np.isfinite(out).all() or not np.allclose(out.sum(1), 1, rtol=0, atol=1e-12):
        raise ValueError("invalid held probabilities")
    return out, checks


def fit(root, held, seed, smoke):
    import braindecode
    import deep
    import features
    import protocol
    import run_cv
    import torch
    from source_labels import read_training_labels

    for module in (deep, features, protocol, run_cv):
        if Path(module.__file__).resolve().parent != SOURCE / REV / "experiments":
            raise ValueError(f"unexpected actor source import {module.__name__}")
    if braindecode.__version__ != "1.8.1":
        raise ValueError("pinned model implementation differs")
    configure_runtime(torch)
    data, _ = read_inputs(root)
    fold, tr, te = next(fold for fold in protocol.folds(data, "all_loso") if fold[0] == held)
    subject = data["subject"]
    if not np.array_equal(te, subject == f"Dreyer2026Proteus/{fold}") or not np.array_equal(tr, ~te):
        raise ValueError("unexpected source fold")
    source, test = np.flatnonzero(tr), np.flatnonzero(te)
    orders = dict(recording=test, **deliveries(root, test, len(subject)))
    data["y"] = np.full(len(subject), -1, dtype=np.int64)
    data["y"][source] = read_training_labels(root / "data/proteus_windows.npz", source, len(subject))
    data["w"] = np.ones(len(subject))
    run_cv.SEED[0] = seed
    actor = run_cv.CONFIGS[MODEL](data)
    if (actor.arch, actor.align, actor.k, actor.epochs, actor.batch, actor.lr, actor.wd, actor.seed,
        actor.dev, actor.adabn, actor.aug, actor.train_fir, actor.ft, actor.ls, actor.lld, actor.lp,
        actor.crops, actor.net_kw) != (
        "shallow", True, 5.0, EPOCHS, BATCH, 1e-3, 1e-3, seed,
        "cuda", None, False, False, 0, 0.0, 1.0, 0, (), {}
    ):
        raise ValueError("unexpected Shallow EA recipe")
    prior = features.covariances(data["X"][source]).mean(0)
    actor.fit(source)
    if type(actor.net) is not braindecode.models.ShallowFBCSPNet or actor.net.training:
        raise ValueError("unexpected fitted model or mode")
    batches = {name: int(buffer) for name, buffer in actor.net.named_buffers()
               if name.endswith("num_batches_tracked")}
    expected_batches = EPOCHS * math.ceil(len(source) / BATCH)
    if not batches or any(count != expected_batches for count in batches.values()):
        raise ValueError("fitted model did not see the full registered batch schedule")
    if not np.array_equal(actor.G, prior) or (data["y"][test] != -1).any():
        raise ValueError("unexpected alignment prior or held labels")

    state = tensor_identity(actor.net)
    values, checks = {}, {}
    for name, delivery in orders.items():
        values[name], checks[name] = replay(actor, features, test, source, delivery, torch)
    changed = {name: float(np.abs(values[name] - values["recording"]).max()) for name in ORDERS}
    if tensor_identity(actor.net) != state or max(changed.values()) <= 1e-4:
        raise ValueError("frozen state changed or all deliveries reused recording probabilities")
    if held == "P01":
        full0, = checks["full0"].values()
        if full0["first_row"] != 94 or full0["rec_lookup_max_abs"] <= 1e-3:
            raise ValueError("P01 full0 no-op sentinel failed")

    if smoke:
        actor.fit(source)
        if tensor_identity(actor.net) != state:
            raise ValueError("same-fold deterministic refit changed the trained state")
        for name in LAYOUT:
            post, _ = replay(actor, features, test, source, orders[name], torch)
            if not np.array_equal(post, values[name]):
                raise ValueError("same-fold deterministic refit changed held probabilities")

    tag = f"{held}_seed{seed}"
    checkpoint = HERE / f"{tag}_fresh.pt"
    output = HERE / f"{tag}_orders.npz"
    # Serialize the actual fitted state, the source-only prior and the row identity.
    with checkpoint.open("xb") as handle:
        torch.save(dict(model={k: v.cpu() for k, v in actor.net.state_dict().items()},
                        prior=torch.from_numpy(prior), fold=held, seed=seed, revision=REV, state=state), handle)
    with output.open("xb") as handle:
        np.savez_compressed(handle, index=test, orders=np.array(LAYOUT),
                            delivery=np.stack(list(orders.values())), proba=np.stack(list(values.values())))
    receipt = dict(**provenance(), fold=held, seed=seed, source_people=sorted(set(subject[source])),
                   fit_rows_sha=fingerprint(source.astype("<i8")), held_rows_sha=fingerprint(test.astype("<i8")),
                   source_label_sha=fingerprint(data["y"][source]), source_prior_sha=fingerprint(prior),
                   batches=batches, state=state, checkpoint_sha=digest(checkpoint), predictions_sha=digest(output),
                   gpu=torch.cuda.get_device_name(), node=os.environ["SLURMD_NODENAME"],
                   deterministic=True, duplicate_fit_bit_exact=smoke, checks=checks, delta_from_recording=changed)
    write_json(HERE / f"{tag}_receipt.json", receipt)
    print(f"PASS new Shallow EA {held} seed {seed} state {state}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("held", choices=PEOPLE)
    parser.add_argument("seed", type=int, choices=SEEDS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke != ((args.held, args.seed) == ("P10", 0)):
        raise ValueError("P10 seed 0 is the exclusive real-data smoke")
    fit(args.root, args.held, args.seed, args.smoke)
