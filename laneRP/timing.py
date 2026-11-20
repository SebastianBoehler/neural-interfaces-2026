"""Warm single-window execution of two fine-tuned traces, including causal alignment and reset."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from features import spectral_identity
from waveform import zero


def stream(expert, x):
    sums = [np.zeros((41, 41), np.float64) for _ in expert.members]
    for model, _ in expert.members:
        zero(model)
    result = []
    start = time.perf_counter()
    with torch.no_grad(), torch.jit.optimized_execution(False):
        for n, window in enumerate(x, 1):
            wave = window.astype(np.float64)
            c = wave - wave.mean(1, keepdims=True)
            cov = c @ c.T / c.shape[1]
            probs = []
            for i, (model, p) in enumerate(expert.members):
                if bool(p["align"]):
                    sh, prior = float(p["shrink"]), float(p["k"])
                    sums[i] += (1 - sh) * cov + sh * cov.trace() / len(cov) * np.eye(len(cov))
                    val, u = np.linalg.eigh((prior * p["G"] + sums[i]) / (prior+n))
                    window_in = ((u * np.maximum(val, 1e-12)**-.5) @ u.T) @ wave
                else:
                    window_in = wave / float(p["scale"])
                a = torch.as_tensor(window_in, device="cuda", dtype=torch.float32)[None]
                probs.append(model(a)[:, :3].softmax(-1).cpu().numpy()[0])
            result.append(probs)
    return np.array(result), time.perf_counter() - start


def timing(expert, x, names, fingerprint, destination):
    x = x[:100]
    target = expert.predict(x)
    stream(expert, x[:12])
    times, identity = [], []
    for _ in range(3):
        pred, sec = stream(expert, x)
        np.testing.assert_array_equal(pred.mean(1).argmax(1), target.mean(1).argmax(1))
        np.testing.assert_allclose(pred, target, atol=3e-4)
        times.append(sec)
        # Charge the spectral feature and projection on every window, even after the latch.
        t = time.perf_counter()
        proj = np.stack([fingerprint.gate.proj(spectral_identity(v)) for v in x])
        at, name = fingerprint.route(names, proj)
        identity.append(time.perf_counter()-t)
    Path(destination).write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(), n=len(x),
                                expert_seconds=times, gate_seconds=identity, at=at, name=name)))
