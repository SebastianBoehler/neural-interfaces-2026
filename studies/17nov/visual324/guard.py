"""Fresh real-source no-score rendering/frequency/state guards before324labels."""
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
from calibrate import fit
from display import LAYOUTS, VIEWS
from shared import build, check, digest, features, load_data, offline, sha, state_sha


def main():
    root = Path(os.environ['VISUAL_ROOT']).resolve(strict=True)
    torch.set_num_threads(6)
    check(root); d, fit_rows, val, people = load_data()
    display, meta = fit(root, d, fit_rows)
    frequencies, kernel = display.frequencies, display.kernel
    if (len(display.centers) != 14 or abs(float(display.centers[0]) - 1) > 1e-4 or
        abs(float(display.centers[-1]) - 55) > 1e-4 or torch.any(kernel[:, (frequencies <= 0) | (frequencies >= 60)] != 0)):
        raise ValueError('frequency rows or Nyquist support invalid')
    time = torch.arange(480) / 120
    tone_rows = []
    with torch.no_grad():
        for i, center in enumerate(display.centers):
            tone = torch.sin(2 * torch.pi * center * time).reshape(1, 1, 480).repeat(1, 41, 1)
            energy = display.logpower(tone, 41).exp()[:, :, :, 160:320].mean((0, 1, 3))
            peak = int(energy.argmax())
            if abs(peak - i) > 1:
                raise ValueError(f'frequency row inactive/misplaced:{i}->{peak}')
            tone_rows.append(peak)
    models = {v: build(root, v, 'cpu') for v in ('pretrained', 'random')}
    states = {v: state_sha(m) for v, m in models.items()}
    sample_rows = np.array([np.flatnonzero(people == f'P{i:02d}')[0] for i in (4, 5)])
    x = torch.from_numpy(d['X'][sample_rows].copy())
    results = {}; saved_features = {}
    def infer(model, signal, layout, view):
        with torch.no_grad(), offline():
            return features(model, display(signal, layout, view), layout * 35).numpy()
    raw = display.raw_meanpower(x).numpy(); raw_sha = digest(raw)
    for layout in LAYOUTS:
        for view in VIEWS:
            image = display(x[:1], layout, view)
            if image.shape != (1, 3, layout * 14, 490): raise ValueError('image shape failed')
            if view == 'frequency' and float(display.plane(x[:1], layout, view).std(2).mean()) <= 1e-6:
                raise ValueError('frequency rows were duplicated instead of distinct')
            for variant, model in models.items():
                original = infer(model, x[:1], layout, view)
                saved_features[f'{layout}|{view}|{variant}'] = original
                infer(model, x[1:2], layout, view)
                repeated = infer(model, x[:1], layout, view)
                doubled = infer(model, 2 * x[:1], layout, view)
                changed = x[:1].clone(); changed[:, 0, -1] += display.scale
                tail = infer(model, changed, layout, view)
                amp = float(np.max(np.abs(doubled - original))); last = float(np.max(np.abs(tail - original)))
                if not np.array_equal(original, repeated) or min(amp, last) <= 1e-7:
                    raise ValueError('determinism/amplitude/final-sample guard failed')
                results[f'{layout}|{view}|{variant}'] = dict(amplitude=amp, final_sample=last)
            xx = x[:1].clone().requires_grad_(True)
            value = features(models['pretrained'], display(xx, layout, view), layout * 35)[0, 0]
            gradient = torch.autograd.grad(value, xx)[0]
            if not torch.isfinite(gradient).all() or float(gradient.abs().max()) <= 1e-12:
                raise ValueError('display/visual input gradient inactive')
            # Same raw block irrespective of renderer/prior; only append it explicitly later.
            direct = display.direct(x, layout, view).detach().numpy()
            if not np.array_equal(direct[:, -287:], raw): raise ValueError('raw skip changed with display')
    if any(state_sha(m) != states[v] for v, m in models.items()): raise ValueError('frozen model state changed')
    with (root / 'guard_features.npz').open('xb') as out: np.savez(out, **saved_features)
    report = dict(passed=True, labels_opened=False, source=meta, states=states,
        source_windows=sample_rows.tolist(), tone_peak_rows=tone_rows, frequency_centers=display.centers.tolist(),
        filter_sha256=digest(display.kernel.numpy()), raw_skip_sha256=raw_sha, checks=results,
        source_sha256=sha(root / 'source.sha256'), renderer_sha256=sha(root / 'renderer.npz'), feature_guard_sha256=sha(root / 'guard_features.npz'))
    with (root / 'guard.json').open('x') as out: json.dump(report, out, indent=2)
    print('324real-source no-score frequency/render/gradient/state guards passed', flush=True)
    queued = subprocess.check_output(['squeue', '-u', 'gfh527', '-h', '-r', '-o', '%i'], text=True).splitlines()
    if len(queued) + 1 > 244: raise RuntimeError('capacity unavailable;no pilot queued')
    job = subprocess.check_output(['sbatch', '--parsable', f'--dependency=afterok:{os.environ["SLURM_JOB_ID"]}', 'source.sbatch'], cwd=root, text=True).strip().split(';')[0]
    (root / 'continuation.json').write_text(json.dumps(dict(source_pilot_job=job)))
    print(f'queued one A100source-only pilot{job}after passed guard', flush=True)


if __name__ == '__main__': main()
