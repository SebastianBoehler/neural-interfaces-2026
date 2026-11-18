"""Replay F11 on genuine EEG with controlled sensor-layer histories."""

import numpy as np
import torch

from fit import ORDERS, PEOPLE
from input_scale import KINDS, check_behavior, embed, scale_episode, window_power
from raw import RawDecoder


ARMS = ("genuine", "sensor_donor", "sensor_identity", "blank")


def decode_episode(model, eeg, reference, delivered, donor):
    genuine, _ = scale_episode(eeg, reference, delivered)
    donated, _ = scale_episode(eeg, reference, donor)
    torch.testing.assert_close(genuine[:1], eeg[:1], rtol=0, atol=0)
    torch.testing.assert_close(donated[:1], eeg[:1], rtol=0, atol=0)
    z, static = embed(model, genuine)
    writes = model.writes(z, static)
    prior = torch.cat((torch.zeros_like(writes[:1]), writes[:-1].cumsum(dim=0)))
    ages = torch.arange(len(z), device=z.device)
    result = dict(genuine=model.read(z, static, prior, ages), blank=static)
    for key, current in (("sensor_donor", donated), ("sensor_identity", eeg)):
        latent, logits = embed(model, current)
        result[key] = model.read(latent, logits, prior, ages)
    return result


@torch.no_grad()
def score_checkpoint(path, x, people, streams, perms, device="cpu"):
    payload = torch.load(path, weights_only=True, map_location=device)
    held, donor = payload["held"], payload["donor_stream"]
    if (payload["kind"] not in KINDS or held not in PEOPLE
            or donor.split("|", 1)[0] == held or donor not in streams
            or donor != sorted(set(streams[people != held]))[0]):
        raise ValueError("wrong fold or saved donor stream")
    model = RawDecoder().to(device).eval()
    model.load_state_dict(payload["model"], strict=True)
    reference, donor_power = payload["reference"], payload["donor_power"]
    donor_eeg = torch.from_numpy(x[streams == donor]).to(device)
    if (reference.shape != (41,) or reference.dtype != torch.float64 or not torch.isfinite(reference).all()
            or torch.any(reference <= 0) or donor_power.shape != (len(donor_eeg), 41)
            or donor_power.dtype != torch.float64 or not torch.isfinite(donor_power).all()
            or torch.any(donor_power < 0)):
        raise ValueError("invalid fixed source reference or donor")
    torch.testing.assert_close(donor_power, window_power(donor_eeg), rtol=3e-5, atol=1e-8)
    idx = np.flatnonzero(people == held)
    probe = torch.from_numpy(x[idx[:24]]).to(device)
    check_behavior(model, probe, reference, window_power(probe))
    check_behavior(model, probe, reference, donor_power[:len(probe)])
    result = {arm: np.full((len(ORDERS), len(idx), 3), np.nan, np.float32)
              for arm in ARMS}
    age = np.full((len(ORDERS), len(idx)), -1, np.int64)
    for i, order in enumerate(ORDERS):
        perm = perms[order]
        for stream in sorted(set(streams[idx])):
            rows = perm[streams[perm] == stream]
            index = np.searchsorted(idx, rows)
            eeg = torch.from_numpy(x[rows]).to(device)
            donors = donor_power[torch.arange(len(rows), device=device) % len(donor_power)]
            logits = decode_episode(model, eeg, reference, window_power(eeg), donors)
            if i == 0:
                prefix = decode_episode(model, eeg[:9], reference, window_power(eeg[:9]), donors[:9])
                for arm in ARMS:
                    torch.testing.assert_close(prefix[arm], logits[arm][:9], rtol=3e-5, atol=3e-5)
            for arm in ARMS:
                result[arm][i, index] = logits[arm].cpu().numpy()
            age[i, index] = np.arange(len(rows))
    if (age < 0).any() or any(not np.isfinite(logits).all() for logits in result.values()):
        raise ValueError("incomplete F11 held waveform bank")
    for arm in ARMS:
        if not np.array_equal(result[arm][age == 0], result["genuine"][age == 0]):
            raise ValueError("first-window exact identity failed")
        if arm != "genuine" and np.max(np.abs(result[arm] - result["genuine"])) <= 1e-5:
            raise ValueError(f"inactive F11 held history control {arm}")
    return dict(index=idx, age=age, **result)
