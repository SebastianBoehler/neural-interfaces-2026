"""Fail closed unless the real-data CPU smoke validates these exact model sources."""

import hashlib
import json
import math
import os
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    producer = Path(os.environ['S308_WORKSPACE']).resolve(strict=True)
    smoke = Path(os.environ['S308_SMOKE_ROOT']).resolve(strict=True)
    report = json.loads((smoke / 'runs/smoke/report.json').read_text())
    expected = {'linear_structured', 'tanh_structured', 'linear', 'quadratic', 'fourier'}
    if (report.get('no_held_accuracy') is not True or report.get('device') != 0 or
            report.get('outer_person') != 'P01' or report.get('inner_person') != 'P02' or
            set(report.get('arms', {})) != expected or not math.isfinite(report['prefix_max_abs']) or
            not math.isfinite(report['reset_max_abs']) or report['prefix_max_abs'] > 1e-10 or
            report['reset_max_abs'] > 1e-10 or report['library_sha256'] != digest(Path(os.environ['S308_LIB']))):
        raise ValueError('real-data CPU smoke report is incomplete or failed')
    for name, arm in report['arms'].items():
        if (not math.isfinite(arm['activity']) or not math.isfinite(arm['reference_error']) or
                arm['activity'] <= 0 or arm['reference_error'] > 2e-5 or
                set(arm['max_probability_difference']) != {'0'} or
                not math.isfinite(arm['max_probability_difference']['0']) or
                arm['max_probability_difference']['0'] > 2e-5):
            raise ValueError(f'{name}: head activity or CPU parity failed')
        if digest(smoke / 'runs/smoke' / f'{name}.pt') != arm['checkpoint_sha256']:
            raise ValueError(f'{name}: smoke checkpoint changed')
    relative = 'studies/16nov/followups/strong_heads'
    for name in ('heads.py', 'train.py', 'export.py', 'geometry.py', 'split.py', 'native.py', 'causal_native.py'):
        if digest(smoke / relative / name) != digest(producer / relative / name):
            raise ValueError(f'{name}: producer differs from CPU-smoked model')
    for name, sha in report['input_hashes'].items():
        if digest(Path(os.environ['S308_CACHE']) / name) != sha:
            raise ValueError(f'{name}: source input changed after smoke')
    print('real-data CPU smoke and all source/input/checkpoint guards passed; no score', flush=True)


if __name__ == '__main__':
    main()
