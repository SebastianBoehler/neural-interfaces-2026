"""Fit the original owning three-seed Eye ensemble excluding both target people."""

import argparse
import json
import os
from pathlib import Path
from time import monotonic

import numpy as np
import torch

from eye_scope import BASE, DATA, DATA_SHA, OCUNET, PEOPLE, PRODUCER, pin, rows_sha, sha
from fit_eye import fit_actual, prefix_checks
from source_labels import read_training_labels


FIT = BASE / "ni26_friday_return_eye_cal_fit_v1_20261119"
N_SOURCE, N_EXCLUDED, BATCHES = 1512, 1143, 24


def scope():
    pin(DATA, DATA_SHA)
    with np.load(DATA, allow_pickle=False) as z:
        person, visit = z["subject"], z["session"]
    target = np.isin(person, [f"Dreyer2026Proteus/{who}" for who in PEOPLE])
    source, excluded = np.flatnonzero(~target), np.flatnonzero(target)
    if (len(source), len(excluded), len(set(person[source]))) != (N_SOURCE, N_EXCLUDED, 8):
        raise ValueError("the joint participant exclusion scope differs")
    return source, excluded, person, visit


def fit(seed, smoke):
    if ("A100" not in torch.cuda.get_device_name() or torch.__version__ != "2.14.1+cu126"
            or os.environ["SLURM_CPUS_PER_TASK"] != "4"):
        raise ValueError("requires the original capped A100 Eye fit runtime")
    torch.set_num_threads(4)
    start = monotonic()
    idx, excluded, people, visits = scope()
    with np.load(DATA, allow_pickle=False) as z:
        x, channels = z["X"][idx], z["ch_names"]
    if x.dtype != np.float32 or x.shape != (N_SOURCE, 41, 480):
        raise ValueError("real joint source geometry is missing")
    stream = np.char.add(np.char.add(people[idx], "|"), visits[idx])
    y = read_training_labels(DATA, idx, len(people))
    d = dict(X=x, ch_names=channels, stream=stream, y=y)
    path = Path("smoke" if smoke else "fits")
    trace, recipe, updates, loads = fit_actual(d, seed, smoke, path, BATCHES)
    tests = prefix_checks("P01", d, trace, 190)
    trace.chmod(0o444)
    proof = dict(source=sha("fit_source.sha256"), member="eye", seed=seed, smoke=smoke, epochs=recipe["epochs"],
                 updates=updates, source_loads=loads, recipe=sha(PRODUCER), ocular_features=sha(OCUNET),
                 train_rows=rows_sha(idx), excluded_rows=rows_sha(excluded), n_source=len(idx),
                 n_excluded=len(excluded), excluded_people=PEOPLE, excluded_waveforms_used_in_fit=False,
                 excluded_labels_opened=False, trace=sha(trace), export_max_abs=recipe["trace_vs_eager"],
                 tests=tests, backend=torch.cuda.get_device_name(), elapsed_seconds=monotonic() - start)
    receipt = trace.with_suffix(".json")
    with receipt.open("x") as f:
        json.dump(proof, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
    receipt.chmod(0o444)
    print("PASS authentic joint-excluded Eye", seed, sha(receipt), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("seed", type=int, choices=range(3))
    cli.add_argument("--smoke", action="store_true")
    arg = cli.parse_args()
    fit(arg.seed, arg.smoke)
