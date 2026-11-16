"""Validate all ten fold shards before assembling per-arm/order deltas."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get('NI26_WS', Path(__file__).resolve().parents[4])).resolve()
sys.path.insert(0, str(ROOT / 'studies/16nov/followups/geometry'))
from chain import ORDERS, sha, validate_delta  # noqa: E402
from config import ARMS, ARCHIVES  # noqa: E402

PEOPLE = tuple(f'P{i:02d}' for i in range(1, 11))


def validate_record(record, session, n, cap):
    expected = 3 * len(range(32, n, 8))
    if (record.get('session') != session or record.get('windows') != n or record.get('updates') != expected or
            not np.isfinite([record.get('grad_max'), record.get('bias_norm')]).all() or
            record['bias_norm'] > cap + 1e-6 or
            (expected and (record['grad_max'] <= 0 or record['bias_norm'] <= 0))):
        raise ValueError(f'invalid geometry state: {record}')


def main():
    directory = ROOT / 'runs/s295_geometry/shards'
    expected = {f'{p}.npz' for p in PEOPLE}
    found = {p.name for p in directory.glob('*.npz')}
    if found != expected:
        raise ValueError(f'incomplete fold matrix: missing={sorted(expected-found)}, extra={sorted(found-expected)}')
    with np.load(ROOT / 'data/proteus_windows.npz', allow_pickle=False) as z:
        subjects, sessions = z['subject'].copy(), z['session'].copy()
    streams = np.char.add(np.char.add(subjects, '|'), sessions)
    n = len(subjects); source_sha = sha(ROOT / 'source.sha256')
    data_sha = sha(ROOT / 'data/proteus_windows.npz'); perm_sha = sha(ROOT / 'runs/shuffle_perms.npz')
    archive_hash = {name: sha(ROOT / 'runs' / name) for name in sorted(set(ARCHIVES.values()))}
    with np.load(ROOT / 'runs/shuffle_perms.npz', allow_pickle=False) as z:
        perms = {o: z[f'{o[:-1]}|{o[-1]}|all_loso'].astype(np.int64) for o in ORDERS}
    for order, perm in perms.items():
        if not np.array_equal(np.sort(perm), np.arange(n)):
            raise ValueError(f'{order}: invalid permutation')
    combined = {(arm, order): np.full((n, 3), np.nan) for arm in ARMS for order in ORDERS}
    shard_hashes = {}
    for person in PEOPLE:
        file = directory / f'{person}.npz'; before = sha(file)
        mask = np.char.endswith(subjects, '/' + person)
        index, fit = np.flatnonzero(mask), np.flatnonzero(~mask)
        with np.load(file, allow_pickle=False) as z:
            if set(z.files) != {'index', 'metadata'} | {f'{a}|{o}' for a in ARMS for o in ORDERS}:
                raise ValueError(f'{person}: incomplete output keys')
            m = json.loads(str(z['metadata'].item()))
            if (m.get('held') != person or not np.array_equal(z['index'], index) or
                    set(m.get('fit_people', ())) != set(subjects[fit]) or
                    m.get('fit_rows_sha256') != hashlib.sha256(fit.tobytes()).hexdigest() or
                    m.get('source_manifest_sha256') != source_sha or m.get('data_sha256') != data_sha or
                    m.get('permutation_sha256') != perm_sha or m.get('archives_sha256') != archive_hash or
                    m.get('held_accuracy_computed') is not False or m.get('max_rebuilt_diff', np.inf) > 1e-5 or
                    m.get('max_feature_diff', np.inf) > 1e-8):
                raise ValueError(f'{person}: fold/source parity provenance failed')
            for order, perm in perms.items():
                if m['order_sha256'][order] != hashlib.sha256(perm.tobytes()).hexdigest():
                    raise ValueError(f'{person}/{order}: wrong order hash')
                ordered = perm[mask[perm]]
                for arm, (_, cap) in ARMS.items():
                    a = z[f'{arm}|{order}']
                    validate_delta(a, len(index), zero=False)
                    records = m['records'][f'{arm}|{order}']
                    names = list(dict.fromkeys(streams[ordered]))
                    if len(records) != len(names):
                        raise ValueError(f'{person}/{order}/{arm}: session count mismatch')
                    for sess, record in zip(names, records):
                        rows = ordered[streams[ordered] == sess]
                        validate_record(record, str(sess), len(rows), cap)
                        loc = np.searchsorted(index, rows[:32])
                        if not np.array_equal(a[loc], np.zeros((len(loc), 3))):
                            raise ValueError(f'{person}/{order}/{arm}: changed pre-update prefix')
                    combined[arm, order][index] = a
        if sha(file) != before:
            raise ValueError(f'{person}: shard changed while reading')
        shard_hashes[file.name] = before
    if any(not np.isfinite(v).all() for v in combined.values()):
        raise ValueError('a row was not filled')
    out = ROOT / 'runs/s295_geometry/deltas'; out.mkdir(parents=True, exist_ok=True)
    manifest = out.parent / 'treatment_manifest.json'
    names = {f'{a}_{o}.npz' for a in ARMS for o in ORDERS}
    if manifest.exists() or any((out / x).exists() for x in names):
        raise FileExistsError('geometry assembled output collision')
    hashes = {}
    for (arm, order), delta in combined.items():
        path = out / f'{arm}_{order}.npz'
        metadata = {'arm': arm, 'order': order, 'source_manifest_sha256': source_sha,
                    'order_sha256': hashlib.sha256(perms[order].tobytes()).hexdigest()}
        with path.open('xb') as f:
            np.savez_compressed(f, delta=delta, metadata=json.dumps(metadata, sort_keys=True))
        hashes[path.name] = sha(path)
    with manifest.open('x') as f:
        json.dump({'files': hashes, 'shards': shard_hashes, 'source_manifest_sha256': source_sha}, f, indent=2, sort_keys=True)
        f.write('\n')
    print(f'assembled 32 deltas, 10 verified folds, manifest={sha(manifest)}; no score', flush=True)


if __name__ == '__main__':
    main()
