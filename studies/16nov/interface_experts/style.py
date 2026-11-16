"""Current-window-only six-band CAR log-power gate input, without labels."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

BANDS = ((4, 8), (8, 13), (13, 20), (20, 30), (30, 45), (45, 58))
DATA_SHA = '2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


def features(x):
    if x.ndim != 3 or x.shape[1:] != (41, 480) or not np.isfinite(x).all():
        raise ValueError('invalid signal tensor')
    x = x.astype(np.float64); x -= x.mean(axis=1, keepdims=True)
    output = []
    for lo, hi in BANDS:
        sos = butter(4, (lo, hi), btype='band', fs=120., output='sos')
        filtered = sosfiltfilt(sos, x, axis=-1)
        output.append(np.log(np.maximum(np.mean(filtered * filtered, axis=-1), 1e-12)))
    out = np.concatenate(output, axis=1)
    if out.shape != (len(x), 246) or not np.isfinite(out).all():
        raise ValueError('invalid style features')
    return out


def main(mode):
    root = Path(os.environ['STYLE_ROOT']).resolve(strict=True)
    data = Path(os.environ['STYLE_DATA']).resolve(strict=True)
    if sha(data) != DATA_SHA: raise ValueError('data hash mismatch')
    with np.load(data, allow_pickle=False) as z: x = z['X'].copy()
    if x.shape != (2655, 41, 480): raise ValueError('unexpected window inventory')
    a = features(x[:2]); b = features(x[:1])
    if not np.allclose(a[:1], b, rtol=0, atol=1e-12):
        raise ValueError('window feature depends on another window')
    output = root / ('runs/style306/smoke.json' if mode == 'smoke' else 'runs/style306/features.npz')
    if mode not in ('smoke', 'full') or output.exists(): raise ValueError('mode or output collision')
    output.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(data_sha256=DATA_SHA, code_sha256=sha(__file__), bands=BANDS,
                labels_opened=False, per_window_only=True)
    if mode == 'smoke':
        with output.open('x') as f: json.dump(meta, f, sort_keys=True); f.write('\n')
    else:
        if not (root / 'runs/style306/smoke.json').exists(): raise ValueError('missing smoke')
        values = np.concatenate([features(x[i:i + 32]) for i in range(0, len(x), 32)])
        if sha(data) != DATA_SHA: raise ValueError('data changed')
        with output.open('xb') as f:
            np.savez_compressed(f, index=np.arange(len(x), dtype=np.int64), values=values,
                                metadata=json.dumps(meta, sort_keys=True))
    print(mode, output, 'sha256', sha(output), flush=True)


if __name__ == '__main__': main(sys.argv[1])
