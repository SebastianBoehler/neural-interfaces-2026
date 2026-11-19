"""Build matched phase-preserving alpha and beta transforms on real source EEG."""

import torch


N = 480
TOLERANCE = 2e-6


def band_gain(spectrum, donor, lo, hi):
    bins = torch.arange(spectrum.shape[-1], device=spectrum.device)
    frequency = bins / 4
    support = (frequency > lo) & (frequency < hi)
    taper = torch.where(support, (torch.pi * (frequency - lo) / (hi - lo)).sin().square(), 0)
    magnitude = spectrum.abs().square().sum(1).sqrt()
    shifted = torch.gather(magnitude, 1, donor)
    ratio = (shifted / magnitude.clamp_min(1e-12)).clamp(.5, 1.5)
    return 1 + taper * (ratio - 1)


def relative_energy(delta, baseline):
    return delta.square().sum((1, 2), dtype=torch.float64) / baseline


@torch.no_grad()
def perturb(x, directions, signs, sampled):
    if x.ndim != 3 or x.shape[1:] != (41, N) or x.dtype != torch.float32 or len(sampled) < 8:
        raise ValueError("invalid native waveform or source witness rows")
    # The inherited F17 tape supplies both variates. Only its shared direction
    # and matched row masks enter this phase-preserving experiment.
    if signs.shape != (len(x), N // 2 - 1):
        raise ValueError("original training tape shape changed")
    direction = torch.as_tensor(directions, device=x.device)
    result = (torch.empty_like(x), torch.empty_like(x))
    severity, total = [], []
    bins = torch.arange(N // 2 + 1, device=x.device)
    for left in range(0, len(x), 128):
        sl = slice(left, left + 128)
        source = x[sl]
        original = torch.fft.rfft(source)
        donor = (bins[None] + torch.where(direction[sl, None], 2, -2)).clamp(0, N // 2)
        gains = [band_gain(original, donor, *band) for band in ((17, 25), (6, 14))]
        deltas = [torch.fft.irfft(original * (gain[:, None] - 1), n=N) for gain in gains]
        energies = [d.square().sum((1, 2), dtype=torch.float64) for d in deltas]
        norm = source.square().sum((1, 2), dtype=torch.float64)
        target = torch.minimum(torch.minimum(*energies), norm * .07**2)
        if not (target > 0).all() or not torch.isfinite(target).all():
            raise ValueError("source waveform lacks a genuine finite spectral perturbation")
        dose = [(target / energy).sqrt().float() for energy in energies]
        for out, delta, scale in zip(result, deltas, dose, strict=True):
            out[sl] = source + delta * scale[:, None, None]
        rms = [relative_energy(out[sl] - source, norm) for out in result]
        severity.append(float((rms[0] - rms[1]).abs().max()))
        total.append([float(v.sum()) for v in rms])
    centers = [(17, 25), (6, 14)]
    witness = torch.fft.rfft(x[sampled])
    donor = (bins[None] + torch.where(direction[sampled, None], 2, -2)).clamp(0, N // 2)
    errors = []
    for out, (lo, hi) in zip(result, centers, strict=True):
        moved = torch.fft.rfft(out[sampled])
        unchanged = (bins <= lo * 4) | (bins >= hi * 4)
        errors.append(float((moved[..., unchanged] - witness[..., unchanged]).norm() / witness.norm()))
        reduction = ((moved * witness.conj()).real.sum(1) / witness.abs().square().sum(1).clamp_min(1e-24))
        errors.append(float((moved - witness * reduction[:, None]).norm() / witness.norm()))
        if not torch.isfinite(out).all():
            raise ValueError("nonfinite perturbed input")
        if torch.equal(band_gain(witness, donor, lo, hi), torch.ones_like(reduction)):
            raise ValueError("band perturbation did not activate")
    if max(errors + severity) > TOLERANCE or float((result[0][sampled] - result[1][sampled]).abs().max()) <= 1e-4:
        raise ValueError("spectral phase, support or matched displacement check failed")
    means = [sum(row[j] for row in total) / len(x) for j in range(2)]
    return *result, dict(severity_delta=max(severity), beta_severity=means[0], alpha_severity=means[1],
                         spectrum_error=max(errors), checked_rows=len(sampled))
