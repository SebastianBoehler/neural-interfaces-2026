"""Prove the omitted historical EA slots on archived and real held inputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
CFG = {
    "fbcnet_ea_tf": (1, 5, "fbcnet", "c625ecc736a881836a25f85a741ea9147bf698bf"),
    "shallow_ea": (3, 6, "shallow", "82120152ace8731cb262c86c8f00b765f6b81418"),
}
RAW = {"fbcnet_raw_tf": (2, 5), "shallow_raw": (4, 6)}
BASES = ("B2_s0", "B2_s1", "B2_s2", "B2_s012", "B2_s345", "B2_pool")
TOL = 2e-5


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fingerprint(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def read_inputs(root):
    with np.load(root / "data/proteus_windows.npz", allow_pickle=False) as dataset:
        data = {name: dataset[name] for name in ("X", "subject", "session", "split")}
    x, subject = data["X"], data["subject"]
    data["stream"] = np.char.add(np.char.add(subject, "|"), data["session"])
    if x.shape != (2655, 41, 480) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError("unexpected official waveform export")
    if set(subject) != {f"Dreyer2026Proteus/P{i:02d}" for i in range(1, 11)}:
        raise ValueError("unexpected historical held roster")
    with np.load(root / "runs/shuffle_perms.npz", allow_pickle=False) as z:
        full0 = z["full|0|all_loso"]
    if not np.array_equal(np.sort(full0), np.arange(len(subject))):
        raise ValueError("invalid full0 delivery")
    return data, full0


def historical(root):
    stacks = {}
    specs = {name: (v[0], v[1]) for name, v in CFG.items()} | RAW
    with np.load(root / "runs/replay_data/inputs.npz", allow_pickle=False) as bank:
        index = bank["idx|all_loso"]
        if not np.array_equal(index, np.arange(2655)):
            raise ValueError("invalid canonical recording rows")
        bases = [bank[f"{base}|Ps|all_loso"] for base in BASES]
        for name, (slot, seeds) in specs.items():
            members = []
            for seed in range(seeds):
                path = root / f"runs/experiments/probs/{name}-all_loso-s{seed}.npz"
                with np.load(path, allow_pickle=False) as z:
                    order = np.argsort(z["index"])
                    values = z["proba"][order]
                    if (not np.array_equal(z["index"][order], index) or values.shape != (len(index), 3)
                            or not np.isfinite(values).all() or not np.allclose(values.sum(1), 1)):
                        raise ValueError(f"invalid historical recording bank {path.name}")
                    members.append(values)
            if not all(np.array_equal(b[slot], np.mean(members, 0)) for b in bases):
                raise ValueError(f"{name} does not reproduce historical slot {slot}")
            stacks[name] = members
    return stacks


def first_window(actor, features, test, rows):
    x = actor.d["X"]
    first = rows[:1]
    prior = (5 * actor.G + features.covariances(x[first])) / 6
    direct = np.einsum("nij,njt->nit", features.invsqrtm(prior), x[first].astype(np.float64)).astype(np.float32)
    delivered = actor._aligned(rows, True)
    independent = actor._aligned(first, True)
    replay = actor._aligned(rows[:16], True)
    rec = actor._aligned(test, True)[np.searchsorted(test, first)]
    if (first.tolist() != [94] or not np.array_equal(delivered[:1], independent) or
            not np.array_equal(delivered[:16], replay) or not np.allclose(independent, direct, atol=TOL, rtol=0)):
        raise ValueError("shuffled first-window or causal prefix input differs")
    shifted = float(np.abs(rec - direct).max())
    if shifted <= 1e-3:
        raise ValueError("recording-order lookup did not exercise the future-context difference")
    return dict(first_row=int(first[0]), first_recording_age=int(np.searchsorted(test, first[0])),
                first_input_sha=fingerprint(independent), rec_lookup_max_abs=shifted,
                first_direct_max_abs=float(np.abs(independent - direct).max()))


def write_json(path, record):
    with path.open("x") as handle:
        json.dump(record, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def tensor_identity(net):
    h = hashlib.sha256()
    for key, value in sorted(net.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def run(root, name, phase):
    import braindecode
    import deep
    import features
    import protocol
    import run_cv
    import torch

    slot, nseeds, arch, revision = CFG[name]
    for module in (deep, features, protocol, run_cv):
        if Path(module.__file__).resolve().parent != HERE / revision / "experiments":
            raise ValueError(f"unexpected source import {module.__name__}")
    predictions = historical(root)
    data, full0 = read_inputs(root)
    f, tr, te = next(x for x in protocol.folds(data, "all_loso") if x[0] == "P01")
    fit, test = np.flatnonzero(tr), np.flatnonzero(te)
    if not np.array_equal(tr, data["subject"] != "Dreyer2026Proteus/P01") or not np.array_equal(te, ~tr):
        raise ValueError("historical source fold differs")
    rows = full0[te[full0]]
    if len(set(data["stream"][test])) != 1 or not np.array_equal(np.sort(rows), test):
        raise ValueError("unexpected P01 session or permutation")
    run_cv.SEED[0] = 0
    actor = run_cv.CONFIGS[name](data)
    if ((actor.arch, actor.align, actor.k, actor.epochs, actor.lr, actor.wd, actor.batch, actor.seed) !=
            (arch, True, 5.0, 40, 1e-3, 1e-3, 64, 0) or
            (arch == "fbcnet" and (not actor.train_fir or actor.adabn is not None or actor.aug))):
        raise ValueError("unexpected historical member training recipe")
    actor.G = features.covariances(data["X"][fit]).mean(0)
    prior = actor.G.copy()
    if not np.array_equal(prior, actor.C[fit].mean(0)):
        raise ValueError("source-only prior differs from historical row-local covariance")
    receipt = dict(fold=f, seed=0, cfg=name, slot=slot, pooled_seeds=list(range(nseeds)), source=revision,
                   source_manifest=digest(HERE / "source.sha256"), archive_manifest=digest(HERE / "archive.sha256"),
                   fit_rows_sha=fingerprint(fit.astype("<i8")), held_rows_sha=fingerprint(test.astype("<i8")),
                   source_people=sorted(set(data["subject"][fit])), n_fit=len(fit), n_held=len(test),
                   prior_sha=fingerprint(prior), fitted_state_available_in_archive=False, held_labels_opened=False,
                   torch=torch.__version__, braindecode=braindecode.__version__, numpy=np.__version__,
                   **first_window(actor, features, test, rows))
    if phase == "inputs":
        write_json(HERE / f"{name}_inputs.json", receipt)
        print(f"PASS {name} authenticated archive and causal input probe", flush=True)
        return

    from source_labels import read_training_labels

    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name():
        raise ValueError("the original A100 backend is required for reconstruction")
    if not (HERE / f"{name}_inputs.json").exists():
        raise ValueError("label-free archive/input pass required before reconstruction")
    data["y"] = np.full(len(tr), -1, dtype=np.int64)
    data["y"][fit] = read_training_labels(root / "data/proteus_windows.npz", fit, len(tr))
    data["w"] = np.ones(len(tr))
    actor.fit(fit)
    if not np.array_equal(actor.G, prior) or (data["y"][test] != -1).any() or actor.net.training:
        raise ValueError("source training changed the held exclusion or alignment prior")
    if arch == "fbcnet" and not all(p.requires_grad for p in actor.net.spectral_filtering.parameters()):
        raise ValueError("historical FBCNet trainable FIR restriction missing")
    frozen = tensor_identity(actor.net)
    rec = protocol.predict_streams(actor, data, te, proba=True)
    target = predictions[name][0][test]
    if rec.shape != target.shape or not np.isfinite(rec).all():
        raise ValueError("invalid recording inference")
    label_changes = int(np.count_nonzero(rec.argmax(1) != target.argmax(1)))
    max_abs = float(np.abs(rec - target).max())
    passed = label_changes == 0 and max_abs <= TOL
    receipt.update(recording_label_changes=label_changes, recording_max_abs=max_abs, parity=passed,
                   fitted_state_sha=frozen, gpu=torch.cuda.get_device_name())
    if not passed:
        write_json(HERE / f"{name}_recording.json", receipt)
        raise ValueError(f"archived recording mismatch {name}: {label_changes} changed labels, max {max_abs}")
    delivered = actor.predict_proba_stream(rows)
    prefix = actor.predict_proba_stream(rows[:16])
    other = np.flatnonzero(data["stream"] == data["stream"][fit[0]])[:16]
    actor.predict_proba_stream(other)
    reset = actor.predict_proba_stream(rows[:16])
    if (not np.array_equal(prefix, reset) or not np.allclose(delivered[:16], prefix, atol=TOL, rtol=0) or
            not np.array_equal(delivered[:16].argmax(1), prefix.argmax(1)) or tensor_identity(actor.net) != frozen):
        raise ValueError("causal model prefix, reset or frozen-state check failed")
    receipt["full0_prefix_posterior_max_abs"] = float(np.abs(delivered[:16] - prefix).max())
    path = HERE / f"{name}_P01_seed0_full0.npz"
    with path.open("xb") as handle:
        np.savez_compressed(handle, index=test, delivery=rows, proba=delivered[np.argsort(rows)])
    write_json(HERE / f"{name}_recording.json", dict(receipt, prediction_sha=digest(path)))
    print(f"PASS {name} recording parity and causal full0 model", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("family", choices=CFG)
    parser.add_argument("phase", choices=("inputs", "reconstruct"))
    args = parser.parse_args()
    run(args.archive, args.family, args.phase)
