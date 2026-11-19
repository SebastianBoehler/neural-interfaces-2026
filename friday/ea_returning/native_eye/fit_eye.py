"""Run the actual production Eye fit on a source-only returning fold."""

import argparse
import json
import os
from pathlib import Path
import runpy
from time import monotonic

import numpy as np
import torch
from torch.optim.optimizer import register_optimizer_step_post_hook

from eye_scope import DATA, FOLDS, OCUNET, PEOPLE, PRODUCER, sha, source_scope
from source_labels import read_training_labels


def source_only(who):
    source, _, people, visits = source_scope(who)
    with np.load(DATA, allow_pickle=False) as z:
        x, channels = z["X"][source], z["ch_names"]
    if x.shape != (len(source), 41, 480) or x.dtype != np.float32:
        raise ValueError("actual source tensor geometry differs")
    stream = np.char.add(np.char.add(people[source], "|"), visits[source])
    y = read_training_labels(DATA, source, len(people))
    return dict(X=x, ch_names=channels, stream=stream, y=y)


def prefix_checks(who, data, trace, n_own):
    ocular = runpy.run_path(str(OCUNET))["traces"]
    source = data["stream"]
    own = np.flatnonzero(source == f"Dreyer2026Proteus/{who}|1")
    other = np.flatnonzero(source != source[own[0]])[:17]
    if len(own) != n_own or len(other) != 17 or len(set(source[other])) != 1:
        raise ValueError("source ocular prefix uses the wrong session rows")

    def features(rows):
        return ocular(dict(X=data["X"][rows], ch_names=data["ch_names"], stream=source[rows]))

    a, b = own[:17], np.r_[own[:7], other, own[7:17]]
    expected = features(a)
    np.testing.assert_array_equal(expected, features(own)[:17])
    np.testing.assert_array_equal(expected, features(b)[source[b] == source[own[0]]])
    np.testing.assert_array_equal(expected[:1], features(own[:1]))
    m = torch.jit.load(str(trace), map_location="cpu").eval()
    with torch.inference_mode():
        batch = m(torch.from_numpy(expected))
        solo = torch.cat([m(torch.from_numpy(x[None])) for x in expected])
        err = float((batch - solo).abs().max())
        if batch.shape != (17, 3) or not torch.isfinite(batch).all() or not 0 <= err < 1e-5:
            raise ValueError(f"invalid exported owning Eye logits: {err}")
        np.testing.assert_array_equal(solo.argmax(1).numpy(), batch.argmax(1).numpy())
    return dict(prefix_exact=True, interleaved_reset_exact=True, first_window_exact=True,
                exported_batch_single_max_abs=err, source_visits=len(set(source)))


def fit_actual(d, seed, smoke, path, batches):
    owner = runpy.run_path(str(PRODUCER))["eye"]
    source_api = owner.__globals__["protocol"]
    saved = source_api.load
    owner.__globals__["OUT"] = path
    target = path / f"eye_s{seed}.pt"
    if target.exists():
        raise FileExistsError(target)
    steps = loads = 0

    def load(*args, **kwargs):
        nonlocal loads
        if args or kwargs or loads:
            raise ValueError("Eye producer changed its source access")
        loads += 1
        return d

    def update(optimizer, _args, _kwargs):
        nonlocal steps
        if not isinstance(optimizer, torch.optim.AdamW) or any(g["weight_decay"] != .01 for g in optimizer.param_groups):
            raise ValueError("owning Eye optimizer differs")
        steps += 1

    hook = register_optimizer_step_post_hook(update)
    source_api.load = load
    try:
        actual = owner(seed, smoke)
    finally:
        source_api.load = saved
        hook.remove()
    epochs = 1 if smoke else 60
    if (loads != 1 or steps != batches * epochs or not callable(owner)
            or (actual["member"], actual["seed"], actual["file"], actual["epochs"]) != ("eye", seed, target.name, epochs)
            or not 0 <= actual["trace_vs_eager"] < 1e-5):
        raise ValueError("production Eye source scopes, actual updates or export parity differ")
    return target, actual, steps, loads


def run(who, seed, smoke):
    if ("A100" not in torch.cuda.get_device_name() or torch.__version__ != "2.14.1+cu126"
            or os.environ["SLURM_CPUS_PER_TASK"] != "4"):
        raise ValueError("requires the registered authentic A100 fit runtime")
    torch.set_num_threads(4)
    started = monotonic()
    d = source_only(who)
    path = Path("smoke" if smoke else "fits") / who
    target, actual, steps, loads = fit_actual(d, seed, smoke, path, 36)
    epochs = actual["epochs"]
    checks = prefix_checks(who, d, target, FOLDS[who][3])
    target.chmod(0o444)
    _, n, n_held, n_own, view, fit_sha, hold_sha = FOLDS[who]
    proof = dict(fold=who, member="eye", seed=seed, smoke=smoke, epochs=epochs, updates=steps,
                 source=sha("fit_source.sha256"), basis=view, fit_rows_sha256=fit_sha, held_rows_sha256=hold_sha,
                 n_source=n, n_held=n_held, n_own_visit1=n_own, actual_source_loads=loads, original_recipe=sha(PRODUCER),
                 ocular_recipe=sha(OCUNET), held_labels_opened=False, held_waveforms_used_in_fit=False,
                 export_max_abs=actual["trace_vs_eager"], trace=sha(target), tests=checks,
                 gpu=torch.cuda.get_device_name(), elapsed_seconds=monotonic() - started)
    receipt = target.with_suffix(".json")
    with receipt.open("x") as handle:
        json.dump(proof, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    receipt.chmod(0o444)
    print("PASS source-only Eye", who, seed, sha(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("person", choices=PEOPLE)
    parser.add_argument("seed", type=int, choices=range(3))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.person, args.seed, args.smoke)
