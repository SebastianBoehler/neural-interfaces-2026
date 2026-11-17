"""Label-free strict/offline rendering, response, state and B1 timing preflight."""
import argparse
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import torch
from assets import prepare, sha, verify
from data319 import DATA, DATA_SHA, fit_display, load, load_display
from encoder import build, features, offline, state_sha
from render import Display


def main(device):
    root = Path(os.environ['VISUAL_ROOT']).resolve(strict=True)
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == 'cuda' and (not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0)):
        raise ValueError('registered A100 route unavailable;no device substitution')
    if device == 'cpu':
        asset = prepare(root)
    else:
        asset = verify(root)
        if not json.loads((root / 'preflight_cpu.json').read_text())['passed']:
            raise ValueError('passed CPU no-score preflight required')
    if sha(DATA) != DATA_SHA:
        raise ValueError('official tensor hash differs')
    x, subject, names, fit, val = load()
    scale, projection, source = fit_display(root, x, subject, names, fit) if device == 'cpu' else load_display(root, fit)
    rows = np.array([np.flatnonzero(np.char.endswith(subject, f'/P{i:02d}'))[0] for i in (3, 4, 5)])
    example = torch.from_numpy(x[rows].copy())
    display = Display(scale, projection).to(device).eval()
    models = {v: build(root, v, device) for v in ('pretrained', 'random')}
    before = {v: state_sha(m) for v, m in models.items()}
    if before['pretrained'] == before['random']:
        raise ValueError('pretrained and random backbones are identical')
    counts = {v: sum(p.numel() for p in m.parameters()) for v, m in models.items()}
    report = dict(device=device, hardware=torch.cuda.get_device_name(0) if device == 'cuda' else 'Slurm-bound6CPU;see node job',
        torch=torch.__version__, numpy=np.__version__, source=source, source_rows=rows.tolist(),
        data_sha256=DATA_SHA, assets_manifest_sha256=sha(root / 'assets/manifest.json'),
        checkpoint_sha256=asset['files']['assets/dinov2_vits14_pretrain.pth'], parameters=counts,
        state_sha256=before, renderer='fixed arctan;no learned renderer', layouts={}, labels_opened=False)
    first_features = {}
    def infer(variant, layout, sample):
        with torch.no_grad(), offline():
            image = display(sample.to(device), layout)
            return features(models[variant], image, layout * 35).cpu().numpy()
    for layout in (41, 16):
        measurements = {v: [] for v in models}
        for v in models:
            infer(v, layout, example[:1]); infer(v, layout, example[:1])
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        responses = {}
        for v in models:
            initial = infer(v, layout, example[:1]); first_features[f'{layout}|{v}'] = initial
            infer(v, layout, example[1:2] + scale)
            repeated = infer(v, layout, example[:1])
            if not np.array_equal(initial, repeated):
                raise ValueError('prefix/interleaved/reset stateless identity failed')
            doubled = infer(v, layout, 2 * example[:1])
            changed = example[:1].clone(); changed[:, 0, -1] += scale
            last = infer(v, layout, changed)
            amp = float(np.max(np.abs(doubled - initial)))
            tail = float(np.max(np.abs(last - initial)))
            if min(amp, tail) <= 1e-7:
                raise ValueError('amplitude or last-original-sample does not affect visual features')
            responses[v] = dict(amplitude_max_abs=amp, last_sample_max_abs=tail)
        for _ in range(2):
            for v in ('pretrained', 'random', 'random', 'pretrained'):
                if device == 'cuda': torch.cuda.synchronize()
                started = time.perf_counter(); infer(v, layout, example[:1])
                if device == 'cuda': torch.cuda.synchronize()
                measurements[v].append(time.perf_counter() - started)
        cpu_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        report['layouts'][str(layout)] = dict(image=[1, 3, layout * 14, 490], patch_tokens=layout * 35,
            feature_dim=768, latency={v: dict(median=float(np.median(t)), minimum=min(t), maximum=max(t), samples=t)
                for v, t in measurements.items()}, input_response=responses, process_peak_rss_bytes=cpu_peak,
            memory_scope='CPU cumulative process high-water incl setup/data/both models,not isolated forward',
            gpu_peak_allocated=torch.cuda.max_memory_allocated() if device == 'cuda' else None,
            gpu_peak_reserved=torch.cuda.max_memory_reserved() if device == 'cuda' else None)
    # Fixed renderer is differentiable; this is an input-flow guard, not display training.
    xx = example[:1].to(device).requires_grad_(True)
    feature = features(models['pretrained'], display(xx, 16), 560)[0, 0]
    gradient = torch.autograd.grad(feature, xx)[0]
    if not torch.isfinite(gradient).all() or float(gradient.abs().max()) <= 1e-12:
        raise ValueError('invalid visual input gradient')
    report['input_gradient_max'] = float(gradient.abs().max())
    if any(state_sha(m) != before[v] or any(p.grad is not None for p in m.parameters()) for v, m in models.items()):
        raise ValueError('frozen backbone state/parameters changed')
    if device == 'cuda':
        with np.load(root / 'cpu_features.npz') as saved:
            errors = {key: float(np.max(np.abs(a - saved[key]))) for key, a in first_features.items()}
        report['cpu_gpu_feature_max_abs'] = errors
        if max(errors.values()) > .005:
            raise ValueError(f'CPU/A100 source-feature guard failed:{errors}')
    else:
        with (root / 'cpu_features.npz').open('xb') as out:
            np.savez(out, **first_features)
    report['passed'] = True
    report['source_manifest_sha256'] = sha(root / 'source.sha256')
    with (root / f'preflight_{device}.json').open('x') as out:
        json.dump(report, out, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('device', choices=('cpu', 'cuda'))
    main(parser.parse_args().device)
