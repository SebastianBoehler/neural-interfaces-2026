"""Replay frozen §323 S1C on Galvani without opening held labels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import MethodType

import numpy as np
import torch

import data323
import encoder312
import model322
import modes
import session_bn
from bounded_bn import normalize, verify_normalization
from data323 import DATA, DATA_SHA, ORDERS, PEOPLE, ROOT, sha


HERE = Path(__file__).resolve().parent
ORIGIN_SHA = "88437e69b5eaf7ff84a3635df88fe68b38394abbc527629b0f41c7fcc24eb6d5"
ARCHIVE_SHA = "89dc9aa77860b8ba4f06925294565d50eb7cf5874f310d598bdad24b4d04a92d"
ORDER_SHA = "6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f"
ATOL = RTOL = 2e-5


def row_sha(indices):
    return hashlib.sha256(indices.astype("<i8", copy=False).tobytes()).hexdigest()


def origin_report():
    if sha(ROOT / "source323.sha256") != ORIGIN_SHA or sha(ROOT / "whole_results.json") != ARCHIVE_SHA:
        raise ValueError("original §323 source or result identity differs")
    for module, name in ((data323, "data323.py"), (model322, "model322.py"),
                         (encoder312, "encoder312.py"), (modes, "modes.py"),
                         (session_bn, "experiments/session_bn.py")):
        if Path(module.__file__).resolve() != ROOT / name:
            raise ValueError(f"wrong original import {module.__name__}")
    return json.loads((ROOT / "whole_results.json").read_text())


def raw_inputs():
    with np.load(DATA, allow_pickle=False) as z:
        x, subject, session = z["X"], z["subject"].astype(str), z["session"].astype(str)
    if x.shape != (2655, 41, 480) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError("unexpected official waveform")
    people = np.array([value.split("/")[-1] for value in subject])
    if set(people) != set(PEOPLE):
        raise ValueError("original roster differs")
    stream = np.char.add(np.char.add(subject, "|"), session)
    with np.load(ROOT / "runs/shuffle_perms.npz", allow_pickle=False) as z:
        perms = {order: z[f"{order[:-1]}|{order[-1]}|all_loso"] for order in ORDERS}
    for p in perms.values():
        if not np.array_equal(np.sort(p), np.arange(len(x))):
            raise ValueError("delivery order differs")
    return x, subject, people, stream, perms


def locked_fold(report, held, seed, subject, people):
    key = f"{held}_s{seed}"
    directory = ROOT / "output" / key
    fold, checkpoint = directory / "result.npz", directory / "S1.pt"
    if sha(fold) != report["folds"][key]:
        raise ValueError("archived held fold digest differs")
    test, fit = np.flatnonzero(people == held), np.flatnonzero(people != held)
    with np.load(fold, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata"]))
        archived = np.stack([z[f"S1C|{order}"] for order in ORDERS])
        if not np.array_equal(z["index"], test):
            raise ValueError("original recording rows differ")
    expected = dict(held=held, seed=seed, fit_people=sorted(set(subject[fit])),
                    source_rows_sha256=row_sha(fit), held_rows_sha256=row_sha(test),
                    source_manifest_sha256=ORIGIN_SHA, data_sha256=DATA_SHA,
                    permutation_sha256=ORDER_SHA, held_labels_opened=False)
    if any(meta[k] != value for k, value in expected.items()) or len(expected["fit_people"]) != 9:
        raise ValueError("original held exclusion, identity or roster differs")
    if sha(checkpoint) != meta["checkpoints"]["S1"]:
        raise ValueError("original S1 checkpoint digest differs")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (payload["norm"], payload["skip"], payload["seed"], payload["held"], payload["scale"]) != (
            "session", True, seed, held, meta["scale"]):
        raise ValueError("S1 checkpoint does not identify the original model instance")
    if not np.isfinite(meta["scale"]) or meta["scale"] <= 0:
        raise ValueError("invalid original source scale")
    return test, meta, archived, payload


def session_batches(x, stream, test, scale, perms):
    for order in ORDERS:
        rank = np.empty(len(x), dtype=int)
        rank[perms[order]] = np.arange(len(x))
        for session in dict.fromkeys(stream[test]):
            rows = test[stream[test] == session]
            rows = rows[np.argsort(rank[rows], kind="stable")]
            other = test[stream[test] != session][:10]
            yield order, np.searchsorted(test, rows), x[rows] / scale, x[other] / scale


@torch.no_grad()
def run(held, seed, smoke):
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0):
        raise ValueError("original Galvani A100 platform required")
    torch.set_num_threads(6)
    report = origin_report()
    x, subject, people, stream, perms = raw_inputs()
    test, meta, archived, payload = locked_fold(report, held, seed, subject, people)
    model322.SEED = seed
    model = model322.Factor("session", True, "cuda")
    model.load_state_dict(payload["state"])
    model.eval()
    fingerprint = encoder312.fingerprint(model)
    layers = [layer for layer in model.body.modules() if type(layer) is session_bn.SessionBN]
    if len(layers) != 1 or layers[0].k != 20.0 or layers[0].m != 32:
        raise ValueError("expected the original S1C normalization layer")
    layer = layers[0]
    captured = []
    original = np.full_like(archived, np.nan)
    for order, pos, xx, _ in session_batches(x, stream, test, meta["scale"], perms):
        handle = None
        if not captured:
            handle = layer.register_forward_pre_hook(lambda _, args: captured.append(args[0][:10].clone()))
        original[ORDERS.index(order), pos] = modes.probabilities(model, xx, "cuda")
        if handle is not None:
            handle.remove()
    parity = float(np.abs(original - archived).max())
    if not np.allclose(original, archived, atol=ATOL, rtol=RTOL) or not np.array_equal(
            original.argmax(2), archived.argmax(2)):
        raise ValueError(f"archived S1C parity failed: max abs {parity}")
    oracle_error = verify_normalization(layer, captured[0])
    counts = []
    layer.forward = MethodType(lambda self, xx: normalize(self, xx, counts), layer)
    bounded = np.full_like(archived, np.nan)
    active, sessions = 0, 0
    prefix_max = 0.0
    for order, pos, xx, other in session_batches(x, stream, test, meta["scale"], perms):
        counts.clear()
        q = modes.probabilities(model, xx, "cuda")
        if len(counts) != 1:
            raise ValueError("the bounded layer must act once per stream")
        active += int(counts[0])
        i = ORDERS.index(order)
        if not np.array_equal(q[0], original[i, pos[0]]):
            raise ValueError("bounded normalization changed a first-window probability")
        prefix = modes.probabilities(model, xx[:10], "cuda")
        prefix_max = max(prefix_max, float(np.abs(q[:len(prefix)] - prefix).max()))
        if not np.allclose(q[:len(prefix)], prefix, rtol=RTOL, atol=ATOL):
            raise ValueError("future windows changed a bounded prefix")
        if len(other):
            modes.probabilities(model, other, "cuda")
            if not np.array_equal(modes.probabilities(model, xx[:10], "cuda"), prefix):
                raise ValueError("interleaved session changed an earlier prefix")
        bounded[i, pos] = q
        sessions += 1
    if (encoder312.fingerprint(model) != fingerprint or not active or
            not np.isfinite(bounded).all() or not np.allclose(bounded.sum(2), 1, atol=1e-6)):
        raise ValueError("changed frozen state, inactive bounded history or incomplete replay")
    receipt = dict(held=held, seed=seed, data=DATA_SHA, orders=ORDER_SHA, origin=ORIGIN_SHA,
                   archived_output=report["folds"][f"{held}_s{seed}"], checkpoint=meta["checkpoints"]["S1"],
                   actor=sha(HERE / "source.sha256"), fit_people=meta["fit_people"],
                   source_rows=meta["source_rows_sha256"], held_rows=meta["held_rows_sha256"],
                   archived_posterior_max_abs=parity, activation_oracle_max_abs=oracle_error,
                   prefix_posterior_max_abs=prefix_max, first_windows_exact=sessions,
                   active_past_channel_windows=active, device=torch.cuda.get_device_name(0),
                   held_labels_opened=False, original_labels_exact=True, disabled_activations_exact=True)
    if not smoke:
        output = HERE / "folds" / f"{held}_s{seed}.npz"
        tmp = output.with_suffix(".tmp")
        with tmp.open("xb") as handle:
            np.savez_compressed(handle, index=test, original=original.argmax(2).astype(np.int8),
                                bounded=bounded.argmax(2).astype(np.int8),
                                metadata=json.dumps(receipt, sort_keys=True))
        os.link(tmp, output)
        tmp.unlink()
        print(f"{sha(output)}  {output.name}", flush=True)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    print("PASS F13 original parity and bounded normalization", flush=True)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("held", choices=PEOPLE)
    args.add_argument("seed", choices=(0, 1), type=int)
    args.add_argument("--smoke", action="store_true")
    cli = args.parse_args()
    run(cli.held, cli.seed, cli.smoke)
