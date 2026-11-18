"""Fit and causally replay fresh source-excluded Sinc seed-zero checkpoints."""

import argparse
import hashlib
import json
import os

import numpy as np
import torch

from refit_smoke import HERE, ROOT, file_sha, source_data


PEOPLE = tuple(f"P{i:02d}" for i in range(1, 11))
TRUNK = ROOT / "runs/pretrain/sincshallow_ea-m7r-s0.pt"
TOL = 2e-5


def state_hash(net):
    digest = hashlib.sha256()
    for key, weight in sorted(net.state_dict().items()):
        value = weight.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def fit(data, rows):
    import session_bn

    if session_bn.__file__ != str(ROOT / "experiments/session_bn.py"):
        raise ValueError("Sinc SessionBN source differs")
    model = session_bn.make("sincshallow_ea_ptm7r_rot05_sbn", data, 0)
    if (model.arch, model.align, model.epochs, model.dev, model.seed, model.rot,
            model.group, model.causal, model.k_train, model.init.format(seed=0)) != (
            "sincshallow", True, 40, "cuda", 0, .05, 32, False, 20., str(TRUNK)):
        raise ValueError("Sinc source recipe differs")
    model.fit(rows)
    model.set_k(20.)
    streams = data["stream"][rows]
    groups = np.concatenate([np.full(np.count_nonzero(streams == s), i)
                             for i, s in enumerate(dict.fromkeys(streams))])
    if (not np.array_equal(groups, model._row_sess)
            or not np.array_equal(model.C[rows].mean(0), model.G)
            or not model.n_sbn or model.net.training):
        raise ValueError("Sinc source training scope or eval path differs")
    return model


@torch.no_grad()
def evaluate(model, rows):
    x = torch.from_numpy(model._aligned(rows, causal=True)).to(model.dev)
    return torch.softmax(model.net(x)[:, :3].double(), 1).cpu().numpy()


def replay(model, data, held, delivery):
    index = data["stream"]
    sessions = {s: delivery[index[delivery] == s] for s in dict.fromkeys(index[delivery])}
    out = np.full((len(held), 3), np.nan)
    peak = 0.
    for session, rows in sessions.items():
        full = evaluate(model, rows)
        early = evaluate(model, rows[:64])
        other = next((r for s, r in sessions.items() if s != session), None)
        if other is not None:
            evaluate(model, other[:64])
        reset = evaluate(model, rows[:64])
        err = float(np.abs(full[:len(early)] - early).max())
        if err > TOL or not np.array_equal(full[:len(early)].argmax(1), early.argmax(1)) or not np.array_equal(early, reset):
            raise ValueError(f"Sinc prefix or reset differs in {session}")
        peak = max(peak, err)
        out[np.searchsorted(held, rows)] = full
    if not np.isfinite(out).all() or not np.allclose(out.sum(1), 1, rtol=0, atol=1e-10):
        raise ValueError("missing or invalid Sinc posterior")
    return out, peak


def run(task):
    if ("A100" not in torch.cuda.get_device_name() or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
            or torch.__version__ != "2.14.1+cu126"):
        raise ValueError("pinned deterministic A100 runtime differs")
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    person = PEOPLE[task]
    subject = f"Dreyer2026Proteus/{person}"
    data, source, held = source_data(subject)
    with np.load(ROOT / "runs/shuffle_perms.npz", allow_pickle=False) as store:
        order = store["full|0|all_loso"]
    delivery = order[data["subject"][order] == subject]
    np.testing.assert_array_equal(np.sort(delivery), held)
    model = fit(data, source)
    state = state_hash(model.net)
    rows_by_mode = {"full0": delivery, "recording": held}
    evaluated = {name: replay(model, data, held, rows) for name, rows in rows_by_mode.items()}
    if state_hash(model.net) != state or (data["y"][held] != -1).any():
        raise ValueError("Sinc frozen state or held label isolation changed")
    if task == 0:
        repeated = fit(data, source)
        if state_hash(repeated.net) != state:
            raise ValueError("same-worker deterministic Sinc repeat does not match")
        for name, rows in rows_by_mode.items():
            np.testing.assert_array_equal(evaluated[name][0], replay(repeated, data, held, rows)[0])
    path = HERE / "folds" / person
    with path.with_suffix(".pt").open("xb") as handle:
        torch.save(dict(net={k: v.detach().cpu() for k, v in model.net.state_dict().items()},
                        trainable=[k for k, v in model.net.named_parameters() if v.requires_grad],
                        prior=torch.from_numpy(model.G), scale=model.scale, actor="sinc", seed=0, fold=person),
                   handle)
    with path.with_suffix(".npz").open("xb") as handle:
        np.savez_compressed(handle, index=held, delivery=delivery,
                            **{name: values for name, (values, _) in evaluated.items()})
    receipt = dict(fold=person, seed=0, fit_rows_sha256=hashlib.sha256(source.astype("<i8").tobytes()).hexdigest(),
                   predictions_sha256=file_sha(path.with_suffix(".npz")), checkpoint_sha256=file_sha(path.with_suffix(".pt")),
                   source_sha256=file_sha(HERE / "source.sha256"), state_sha256=state, held_labels_opened=False,
                   deterministic_repeat=task == 0, gpu=torch.cuda.get_device_name(),
                   prefix_abs={name: v for name, (_, v) in evaluated.items()})
    with path.with_suffix(".json").open("x") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"PASS fresh Sinc {person}, {len(held)} held windows", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=int, choices=range(10))
    run(parser.parse_args().task)
