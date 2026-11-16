"""Native held-person head fit and causal member streams; never score labels."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(os.environ['NI26_WS']).resolve(strict=True)
os.environ['NI26_ROOT'] = str(ROOT)
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(ROOT / 'experiments'), str(ROOT / 'studies/16nov/followups/geometry')]
import protocol  # noqa: E402
from bridge import Native  # noqa: E402
from config import ARCHIVES  # noqa: E402

ORDERS = ('within0', 'within1', 'within2', 'within3', 'full0', 'full1', 'runs0', 'runs1')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main(task):
    if not 0 <= task < 10:
        raise ValueError('held-person task must be 0–9')
    held = f'P{task + 1:02d}'
    target = ROOT / f'runs/s301_native_shards/{held}.npz'
    if target.exists():
        raise FileExistsError(target)
    refroot = Path(os.environ['S301_ALL_REFERENCES']).resolve(strict=True)
    manifest = json.loads((refroot / 'manifest.json').read_text())
    expected_names = {f'P{i:02d}.npz' for i in range(1, 11)}
    if set(manifest['files']) != expected_names or manifest['held_accuracy_computed'] is not False:
        raise ValueError('incomplete training-only reference heads')
    if manifest['source_sha256'] != '2e9833fe9adfe9385ed80996b00a1cc4f890a4ff4903bb7d96fe9b422299f227':
        raise ValueError('wrong frozen reference source')
    if digest(refroot / f'{held}.npz') != manifest['files'][f'{held}.npz']:
        raise ValueError('reference head checksum mismatch')
    with np.load(refroot / f'{held}.npz', allow_pickle=False) as z:
        fixed = {k: np.ascontiguousarray(z[k]) for k in z.files}
    d = protocol.load()
    folds = [(tr, te) for name, tr, te in protocol.folds(d, 'all_loso') if name == held]
    if len(folds) != 1:
        raise ValueError('expected one held-person fold')
    tr, te = folds[0]
    if not np.array_equal(fixed['fit_rows'], np.flatnonzero(tr)):
        raise ValueError('reference training indices differ')
    parity_sha = digest(Path(os.environ['S301_PARITY_REPORT']))
    core_sha = digest(Path(os.environ['NATIVE_CORE_LIB']))
    train_sha = digest(Path(os.environ['NATIVE_TRAIN_LIB']))
    priors, counts, row_parts, feature_parts, saved = [], [], [], [], None
    shard_hashes = {}
    for i in range(10):
        person = f'P{i + 1:02d}'
        path = ROOT / f'runs/s301_native_inputs/{person}.npz'
        shard_hashes[path.name] = digest(path)
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata'].item()))
            rows = z['rows'].copy()
            if (meta['person'] != person or meta['source_sha256'] != digest(ROOT / 'producer.sha256') or
                    meta['parity_report_sha256'] != parity_sha or meta['native_core_sha256'] != core_sha or
                    meta['native_train_sha256'] != train_sha or meta['held_accuracy_computed'] is not False or
                    meta['rows_sha256'] != hashlib.sha256(rows.tobytes()).hexdigest()):
                raise ValueError(f'{person}: native feature provenance mismatch')
            if np.any(rows < 0) or np.any(rows >= len(tr)) or not len(rows):
                raise ValueError('invalid person row indices')
            priors.append(z['prior'].copy()); counts.append(len(rows))
            if i == task:
                saved = {'rows': rows, 'cov': z['cov'].copy(), 'slow': z['slow'].copy()}
            else:
                row_parts.append(rows); feature_parts.append(z['features'].copy())
    index = np.concatenate(row_parts)
    position = np.argsort(index)
    if not np.array_equal(index[position], fixed['fit_rows']) or saved is None or not np.array_equal(saved['rows'], np.flatnonzero(te)):
        raise ValueError('native train/held rows differ from exact fold')
    train = np.ascontiguousarray(np.concatenate(feature_parts)[position])
    y = np.ascontiguousarray(d['y'][fixed['fit_rows']], dtype=np.int64)
    native = Native()
    prior = native.combine(np.ascontiguousarray(priors), np.asarray(counts, dtype=np.int64), task)
    standardized = native.standardize(train, fixed['mean'], fixed['scale'])
    ref_theta = np.ascontiguousarray(np.column_stack((fixed['coef'], fixed['intercept'])))
    ref_loss, _ = native.objective(standardized, y, ref_theta)
    started = time.monotonic()
    theta, loss, gradmax, calls = native.fit(standardized, y)
    if not np.isfinite([ref_loss, loss, gradmax]).all() or abs(loss - ref_loss) > 1e-5 or gradmax > 1e-5 or calls <= 0:
        raise ValueError(f'native head did not match the registered objective: {loss=} {ref_loss=} {gradmax=} {calls=}')
    coef = np.ascontiguousarray(theta[:, :-1]); intercept = np.ascontiguousarray(theta[:, -1])
    output = {'rows': saved['rows']}
    member_differences = {}
    with np.load(ROOT / 'runs/shuffle_perms.npz', allow_pickle=False) as z:
        perms = {o: z[f'{o[:-1]}|{o[-1]}|all_loso'].astype(np.int64) for o in ORDERS}
    for order, perm in perms.items():
        if not np.array_equal(np.sort(perm), np.arange(len(tr))):
            raise ValueError('invalid pinned delivery order')
        selected = perm[te[perm]]
        all_features = np.empty((len(saved['rows']), 44239), np.float64)
        for session in dict.fromkeys(d['stream'][selected]):
            rows = selected[d['stream'][selected] == session]
            local = np.searchsorted(saved['rows'], rows)
            all_features[local] = native.causal(prior, np.ascontiguousarray(saved['cov'][local]),
                                                np.ascontiguousarray(saved['slow'][local]))
        p_fixed = native.predict(all_features, fixed['mean'], fixed['scale'], fixed['coef'], fixed['intercept'])
        p_train = native.predict(all_features, fixed['mean'], fixed['scale'], coef, intercept)
        member_differences[order] = float(np.max(np.abs(p_train - p_fixed)))
        with np.load(ROOT / 'runs' / ARCHIVES[order], allow_pickle=False) as z:
            archive = z[f'{order}|all_loso'][saved['rows']]
        mismatch = float(np.max(np.abs(p_fixed - archive)))
        if mismatch > 1e-5 or np.any(p_fixed.argmax(1) != archive.argmax(1)):
            raise ValueError(f'{held}/{order}: fixed reconstructed-core parity failed: {mismatch}')
        output[f'Nfixed|{order}'] = p_fixed - archive
        output[f'Ntrain|{order}'] = p_train - archive
        print(f'{held} {order}: native causal streams prepared; no score', flush=True)
    metadata = {'held': held, 'source_sha256': digest(ROOT / 'producer.sha256'), 'input_sha256': shard_hashes,
                'reference_sha256': digest(refroot / f'{held}.npz'), 'parity_sha256': parity_sha,
                'data_sha256': digest(ROOT / 'data/proteus_windows.npz'),
                'permutation_sha256': digest(ROOT / 'runs/shuffle_perms.npz'),
                'archives_sha256': {name: digest(ROOT / 'runs' / name) for name in sorted(set(ARCHIVES.values()))},
                'native_fit_loss': loss, 'reference_loss': ref_loss, 'max_gradient': gradmax,
                'coefficient_max_abs_difference': float(np.max(np.abs(coef - fixed['coef']))),
                'intercept_max_abs_difference': float(np.max(np.abs(intercept - fixed['intercept']))),
                'native_trained_vs_fixed_probability_max_abs': member_differences,
                'objective_evaluations': calls, 'fit_and_stream_seconds': time.monotonic() - started,
                'held_accuracy_computed': False}
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as f:
        np.savez(f, **output, metadata=json.dumps(metadata, sort_keys=True))
    print(f'{held}: native fit and all eight orders complete; no score', flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: fit.py HELD_PERSON_INDEX')
    main(int(sys.argv[1]))
