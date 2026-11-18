"""Scale centered EEG using only earlier windows' per-channel power."""

import hashlib

import numpy as np
import torch


KINDS = ("raw_scale_genuine", "raw_scale_donor")
PSEUDOWINDOWS = 20


def window_power(eeg):
    centered = eeg - eeg.mean(dim=-1, keepdim=True)
    return centered.square().mean(dim=-1, dtype=torch.float64)


def fit_reference(power, weights):
    if power.ndim != 2 or weights.shape != (len(power),) or torch.any(weights <= 0):
        raise ValueError("invalid source weights for fitting sensor power")
    mass = weights.double()
    reference = (power * (mass / mass.sum())[:, None]).sum(dim=0)
    if not torch.isfinite(reference).all() or torch.any(reference <= 0):
        raise ValueError("source sensor reference needs finite positive power")
    return reference


def power_hash(power):
    return hashlib.sha256(np.asarray(power.detach().cpu().numpy(), dtype="<f8").tobytes()).hexdigest()


def scale_gain(reference, earlier_power, age):
    prior = PSEUDOWINDOWS * reference
    estimate = (earlier_power + prior) / (age + PSEUDOWINDOWS)
    return (reference / estimate).sqrt().clamp(.5, 2)


def scale_centered(eeg, gain):
    mean = eeg.mean(dim=-1, keepdim=True)
    return (eeg - mean) * gain.to(eeg.dtype).unsqueeze(-1) + mean


def scale_episode(eeg, reference, delivered_power):
    if delivered_power.shape != eeg.shape[:2] or reference.shape != (eeg.size(1),):
        raise ValueError("sensor power history does not cover the episode")
    previous = delivered_power[:-1].cumsum(dim=0)
    age = torch.arange(1, len(eeg), device=eeg.device).unsqueeze(-1)
    gain = scale_gain(reference, previous, age)
    # Return the original first row bitwise, avoiding center-and-restore roundoff.
    scaled = torch.cat((eeg[:1], scale_centered(eeg[1:], gain)))
    return scaled, torch.cat((torch.ones_like(reference[None]), gain))


def scale_window(eeg, reference, earlier_power, age):
    if age == 0:
        return eeg
    return scale_centered(eeg, scale_gain(reference, earlier_power, age))


def embed(model, eeg):
    latent = model.encode(model.prepare(eeg))
    return latent, model.static(latent)


@torch.no_grad()
def check_behavior(model, eeg, reference, delivered_power):
    scaled, gains = scale_episode(eeg, reference, delivered_power)
    expected_first = torch.ones_like(gains[0])
    torch.testing.assert_close(gains[0], expected_first, rtol=0, atol=0)
    torch.testing.assert_close(scaled[:1], eeg[:1], rtol=0, atol=0)
    if not torch.isfinite(scaled).all() or not torch.any((gains - 1).abs() > 1e-4):
        raise ValueError("sensor scaling is inactive or nonfinite")
    torch.testing.assert_close(scaled.mean(-1), eeg.mean(-1), rtol=3e-5, atol=3e-5)
    latent, static = embed(model, scaled)
    writes = model.writes(latent, static)
    expected = model.sequence(latent, static, writes)
    torch.testing.assert_close(expected[0], static[0], rtol=0, atol=0)
    total = torch.zeros_like(reference)
    actor_state = None
    results = []
    for i, window in enumerate(eeg.split(1)):
        actual = scale_window(window, reference, total, i)
        torch.testing.assert_close(actual, scaled[i:i+1], rtol=3e-5, atol=3e-5)
        logits, actor_state = model.step(actual, actor_state)
        results.append(logits)
        total = total + delivered_power[i]
    torch.testing.assert_close(torch.cat(results), expected, rtol=3e-5, atol=3e-5)
    reset, _ = model.step(scale_window(eeg[:1], reference, torch.zeros_like(total), 0))
    torch.testing.assert_close(reset, results[0], rtol=0, atol=0)
    prefix = min(9, len(eeg))
    truncated, _ = scale_episode(eeg[:prefix], reference, delivered_power[:prefix])
    torch.testing.assert_close(truncated, scaled[:prefix], rtol=3e-5, atol=3e-5)
    z, logits = embed(model, truncated)
    actual = model.sequence(z, logits, model.writes(z, logits))
    torch.testing.assert_close(actual, expected[:prefix], rtol=3e-5, atol=3e-5)
