"""Independently fit original F25 hard, conventional and multiresolution on A100."""
import argparse
from pathlib import Path

import braindecode
import numpy as np
import torch

from anchors import DATA_SHA, ORIGIN_SHA, PEOPLE, RUNTIME, row_sha, sha, split
from encoder312 import fingerprint
import model322
from modulation import BASE, KINDS, MAX_RATIO, MAX_RESERVED, architecture, trainable
import probe
from produce_distill import publish
from recipe_f25a import BD, CANARY_SHA, CPU_CUDA, F25, F25_ACTOR, GPU, ROOT, TORCH, WORK, anchors, canary, historical, tag
from source_labels import read_training_labels
from source_probe import timing
from source_targets import normalizer


def run(held, seed, smoke):
    numeric = dict(cudnn=torch.backends.cudnn.version(), cudnn_tf32=torch.backends.cudnn.allow_tf32,
                   matmul_tf32=torch.backends.cuda.matmul.allow_tf32)
    flags = canary()
    if (torch.cuda.get_device_name() != GPU or str(torch.__version__) != TORCH or braindecode.__version__ != BD
            or Path(probe.__file__).resolve() != F25 / "probe.py"
            or Path(model322.__file__).resolve() != ROOT / "model322.py"
            or any(numeric[k] != flags[k] for k in CPU_CUDA)):
        raise ValueError("pinned A100 hardware, source or fit flags differ")
    torch.set_num_threads(2)
    path = ROOT / "data/proteus_windows.npz"
    if sha(path) != DATA_SHA:
        raise ValueError("original unmodified official waveform input required")
    with np.load(path, allow_pickle=False) as z:
        eeg, subject, visit = (z[k] for k in ("X", "subject", "session"))
    if eeg.shape != (2655, 41, 480) or eeg.dtype != np.float32 or not np.isfinite(eeg).all():
        raise ValueError("unexpected original waveform shape or values")
    rows, stream, people = split(subject, visit, held)
    meta, batches, metadata = historical(held, seed, smoke, rows, stream, people)
    checks, _ = anchors(held, seed, smoke)
    anchor = checks["bare"][1]
    y = read_training_labels(path, rows, len(eeg))
    x, y, scale, mu, sd, weight, observed = model322.setup(dict(X=eeg[rows], y=y, stream=stream),
                                                           np.arange(len(rows)), "cuda", epochs=meta["epochs"])
    flat = [b for ep in batches for b in ep]
    if (scale != anchor["scale"] or not np.isclose(scale, meta["scale"], atol=0, rtol=2e-6)
            or any(not np.array_equal(a, b) for a, b in zip(flat, (r for ep in observed for r in ep), strict=True))):
        raise ValueError("original matched A100 source scale or batches differ")
    fixed = dict(held=held, seed=seed, epochs=meta["epochs"], scale=scale, initial=anchor["initial"],
                 data=DATA_SHA, origin=ORIGIN_SHA, runtime=RUNTIME, source_rows=row_sha(rows), source_people=people,
                 batches=meta["batches"], historical_metadata=metadata, h100_state=meta["state"],
                 h100_weights_loaded=False, held_labels_opened=False, actor=sha(WORK / "actor.sha256"),
                 f25=F25_ACTOR, canary=CANARY_SHA, device=GPU, torch=TORCH, braindecode=BD, **numeric,
                 bare_checkpoint=anchor["checkpoint"], bare_state=anchor["state"],
                 references={kind: c[2] for kind, c in checks.items()}, train_rows_seen=sum(map(len, flat)),
                 diagnostic_rows=row_sha(rows[flat[0][:32]]), rng_isolated=True, zero_identity=True,
                 q_gradients_nonzero=0)
    if any(anchor[k] != fixed[k] for k in ("held", "seed", "epochs", "initial", "source_rows", "source_people",
                                           "batches", "historical_metadata", "h100_state")):
        raise ValueError("independent A100 source scope differs from exact original metadata")
    reference = dict(meta, state=anchor["state"])
    sample, states = x[flat[0][:32]], set()
    for kind in KINDS:
        model322.SEED = seed
        model = model322.Factor("session", False, "cuda")
        normalizer(model)
        if fingerprint(model) != fixed["initial"] or trainable(model) != BASE:
            raise ValueError("original native initialization or geometry differs")
        layer = probe.install(model, kind, sample)
        stats = probe.fit(model, layer, (x, y, mu, sd, weight, batches), reference)
        proof = {} if layer is None else probe.diagnostics(model, layer, sample, kind)
        peak = torch.cuda.max_memory_reserved()
        if peak > MAX_RESERVED or stats["state"] in states or fingerprint(model) != stats["state"]:
            raise ValueError("fitted state or reservation gate failed")
        states.add(stats["state"])
        if smoke:
            elapsed = timing(model, sample)
            if kind == "hard":
                hard_ms = elapsed
            if not 0 < elapsed <= hard_ms * MAX_RATIO:
                raise ValueError("original paired warm source runtime gate failed")
            proof.update(warm_ms=elapsed, hard_ms=hard_ms)
        target = WORK / "checkpoints" / f"{kind}_{tag(held, seed, smoke)}.pt"
        publish(target, model, dict(fixed, kind=kind, peak_reserved=peak, **architecture(kind), **stats, **proof))
        if kind == "hard" and sha(target) != anchor["checkpoint"]:
            raise ValueError("new A100 hard fit differs byte-for-byte from both independent controls")
        target.chmod(0o444)
        target.with_suffix(".json").chmod(0o444)
    print("PASS F25a matched A100 source", held, seed, smoke, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("held", choices=PEOPLE)
    parser.add_argument("seed", type=int, choices=(0, 1))
    parser.add_argument("--smoke", action="store_true")
    a = parser.parse_args()
    run(a.held, a.seed, a.smoke)
