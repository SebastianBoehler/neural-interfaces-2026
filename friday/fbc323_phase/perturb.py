"""Build RMS-matched rigid and dispersive perturbations on source EEG only."""
import hashlib
from types import MethodType

import numpy as np
import torch


SAMPLES = 480
SHIFT = 2
SEVERITY_TOL = 2e-6


def random_tape(batches, n, held, seed):
    rng = np.random.default_rng(17017 + 100 * int(held[1:]) + seed)
    signs = 2 * rng.integers(0, 2, size=(n, SAMPLES // 2 - 1), dtype=np.int8) - 1
    direction = rng.integers(0, 2, size=n, dtype=np.int8).astype(bool)
    digest = hashlib.sha256(signs.tobytes() + direction.tobytes())
    masks = []
    for rows in (r for epoch in batches for r in epoch):
        if len(rows) % 4:
            raise ValueError("the original batch cannot take an exact quarter augmentation")
        mask = np.zeros(len(rows), dtype=bool)
        mask[rng.permutation(len(rows))[:len(rows) // 4]] = True
        digest.update(mask.tobytes())
        masks.append(mask)
    return signs, direction, masks, digest.hexdigest()


@torch.no_grad()
def perturb(x, directions, signs, sampled):
    if x.ndim != 3 or x.shape[1:] != (41, SAMPLES) or x.dtype != torch.float32 or len(sampled) < 8:
        raise ValueError("expected official raw shape and eight perturbed smoke rows")
    xa = torch.fft.rfft(x, dim=-1)
    interior = xa[:, :, 1:-1]
    weights = (interior.real.square() + interior.imag.square()).sum(1).double()
    total = weights.sum(1)
    if (total <= 0).any():
        raise ValueError("source row lacks interior spectral power")
    bins = torch.arange(1, SAMPLES // 2, dtype=torch.float64, device=x.device)
    rigid_cos = torch.cos(2 * torch.pi * (SHIFT / SAMPLES) * bins)
    common_cos = ((weights * rigid_cos).sum(1) / total).clamp(-1, 1)
    common_sin = (1 - common_cos.square()).clamp_min(0).sqrt()

    phase_sign = torch.as_tensor(signs, device=x.device)
    re = common_cos.float()[:, None].expand_as(phase_sign)
    im = common_sin.float()[:, None] * phase_sign
    rotation = torch.complex(re, im)
    transformed = xa.clone()
    transformed[:, :, 1:-1] = interior * rotation[:, None, :]
    if not torch.equal(transformed[:, :, (0, -1)], xa[:, :, (0, -1)]):
        raise ValueError("finite Fourier rotation changed DC or Nyquist")
    phase = torch.fft.irfft(transformed, n=SAMPLES, dim=-1)
    direction = torch.as_tensor(directions, device=x.device)
    jitter = torch.where(direction[:, None, None], x.roll(SHIFT, -1), x.roll(-SHIFT, -1))

    base = x.square().sum((1, 2), dtype=torch.float64)
    distortion = [(out - x).square().sum((1, 2), dtype=torch.float64) / base for out in (jitter, phase)]
    max_delta = float((distortion[0] - distortion[1]).abs().max())
    if (max_delta > SEVERITY_TOL or any(not torch.isfinite(t).all() or (t <= 0).any() for t in distortion)
            or float((jitter - phase).abs().max()) <= 1e-4):
        raise ValueError(f"unmatched, nonfinite or inactive perturbation: max normalized squared error {max_delta}")
    reference = xa[sampled].transpose(1, 2)
    spectrum = reference[..., :, None] * reference[..., None, :].conj()
    power = xa[sampled].abs().norm()
    cross, endpoints = [], []
    for out in (jitter, phase):
        rerun = torch.fft.rfft(out[sampled], dim=-1)
        freq = rerun.transpose(1, 2)
        observed = freq[..., :, None] * freq[..., None, :].conj()
        cross.append(float((observed - spectrum).norm() / spectrum.norm()))
        endpoints.append(float((rerun[..., (0, -1)] - xa[sampled][..., (0, -1)]).abs().norm() / power))
    if max(cross + endpoints) > 2e-6:
        raise ValueError(f"the observed Fourier transform changed cross-spectra or endpoints: {cross}/{endpoints}")
    return jitter, phase, dict(severity_delta=max_delta, jitter_severity=float(distortion[0].mean()),
         phase_severity=float(distortion[1].mean()), max_phase_angle=float(common_cos.acos().max()),
         spectrum_error=max(cross), endpoint_error=max(endpoints), checked_rows=len(sampled))


class Feed:
    """Apply the fixed source tape at the original Factor forward boundary."""
    def __init__(self, alternate, batches, masks):
        self.alternate = alternate
        self.rows = [r for epoch in batches for r in epoch]
        self.masks = [torch.as_tensor(m, device=alternate.device) for m in masks]
        self.calls = 0

    def install(self, model, native, original):
        def forward(actual, current):
            i = self.calls
            if not actual.training or not 0 <= i < len(self.rows):
                raise ValueError("unexpected source augmentation call")
            rows, mask = self.rows[i], self.masks[i]
            augmented = torch.where(mask[:, None, None], self.alternate[rows], current)
            if i == 0 and (not torch.equal(current, original[rows]) or not torch.equal(
                    augmented[~mask], current[~mask]) or torch.equal(augmented[mask], current[mask])):
                raise ValueError("augmentation did not reach only the original selected source rows")
            self.calls += 1
            return native(augmented)
        return MethodType(forward, model)
