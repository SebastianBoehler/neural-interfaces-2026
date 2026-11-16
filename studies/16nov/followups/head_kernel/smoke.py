"""Build the exact P01 source matrix and validate two CPU gradients without scoring."""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = Path(os.environ['S311_BASE']).resolve(strict=True)
sys.path[:0] = [str(HERE), str(BASE / 'experiments'), str(BASE / 'studies/16nov/followups/native_core')]
import protocol  # noqa: E402
from bridge import Native  # noqa: E402
from kernel import Kernel  # noqa: E402


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    out = Path('runs/s311_smoke')
    out.mkdir(parents=True, exist_ok=False)
    d = protocol.load()
    folds = [(tr, te) for held, tr, te in protocol.folds(d, 'all_loso') if held == 'P01']
    if len(folds) != 1:
        raise ValueError('P01 source fold unavailable')
    tr, te = folds[0]
    parts, rows, hashes = [], [], {}
    source_sha = digest(BASE / 'producer.sha256')
    for p in range(2, 11):
        path = Path(os.environ['S311_CACHE']) / f'P{p:02d}.npz'
        hashes[path.name] = digest(path)
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata'].item()))
            r = z['rows'].copy()
            if (meta['person'] != f'P{p:02d}' or meta['source_sha256'] != source_sha or
                    meta['held_accuracy_computed'] is not False or np.any(te[r])):
                raise ValueError('cached source-person provenance mismatch')
            rows.append(r)
            parts.append(z['features'].copy())
    index = np.concatenate(rows)
    position = np.argsort(index)
    if not np.array_equal(index[position], np.flatnonzero(tr)):
        raise ValueError('source rows differ from the P01 nine-person fold')
    raw = np.ascontiguousarray(np.concatenate(parts)[position])
    del parts
    y = np.ascontiguousarray(d['y'][index[position]], dtype=np.int64)
    reference = Path(os.environ['S311_REFERENCES']) / 'P01.npz'
    manifest = json.loads((reference.parent / 'manifest.json').read_text())
    if digest(reference) != manifest['files']['P01.npz']:
        raise ValueError('reference head checksum mismatch')
    with np.load(reference, allow_pickle=False) as z:
        if not np.array_equal(z['fit_rows'], np.flatnonzero(tr)):
            raise ValueError('reference head source rows differ')
        mean, scale = np.ascontiguousarray(z['mean']), np.ascontiguousarray(z['scale'])
        theta = np.ascontiguousarray(np.column_stack((z['coef'], z['intercept'])))
    x = Native().standardize(raw, mean, scale)
    if not np.allclose(x, (raw - mean) / scale, atol=1e-12, rtol=1e-12):
        raise ValueError('native and reference source standardization differ')
    del raw
    kernel = Kernel(os.environ['S311_LIB'])
    checks = {}
    for name, xx, yy in (('first32', x[:32], y[:32]), ('full', x, y)):
        ref_loss, ref_grad = kernel.objective(xx, yy, theta, analytical=False)
        loss, grad = kernel.objective(xx, yy, theta, analytical=True)
        loss_diff = abs(loss - ref_loss)
        grad_diff = float(np.max(np.abs(grad - ref_grad)))
        if not np.isfinite([loss, ref_loss, loss_diff, grad_diff]).all() or max(loss_diff, grad_diff) > 1e-9:
            raise ValueError(f'{name}: analytical objective/gradient mismatch')
        checks[name] = {'loss': loss, 'reference_loss': ref_loss, 'loss_abs_diff': loss_diff,
                        'gradient_max_abs_diff': grad_diff}
    matrix = out / 'source_matrix.npz'
    with matrix.open('xb') as f:
        np.savez(f, x=x, y=y, theta=theta)
    report = {'checks': checks, 'matrix_sha256': digest(matrix), 'source_manifest_sha256': digest(Path('source.sha256')),
              'library_sha256': digest(Path(os.environ['S311_LIB'])), 'reference_sha256': digest(reference),
              'source_cache_sha256': hashes, 'rows_sha256': hashlib.sha256(index[position].tobytes()).hexdigest(),
              'shape': list(x.shape), 'held_accuracy_computed': False}
    with (out / 'report.json').open('x') as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write('\n')
    print('real source mini/full objective and gradient checks passed; no held accuracy', flush=True)


if __name__ == '__main__':
    main()
