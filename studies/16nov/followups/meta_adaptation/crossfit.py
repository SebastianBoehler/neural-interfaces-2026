"""Fit one §307 8-person anchor and emit source-person causal features."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler

ROOT = Path(os.environ['NI26_WS']).resolve(strict=True)
CACHE = Path(os.environ['S307_CACHE']).resolve(strict=True)
os.environ['NI26_ROOT'] = str(ROOT)
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT / 'experiments'), str(ROOT / 'studies/16nov/followups/native_core')]
import protocol
from bridge import Native
from ids import person_ids

ORDERS = ('within0', 'full0', 'runs0')


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def fit_native(native, features, labels):
    theta, loss, grad, calls = native.fit(features, labels)
    if not np.isfinite(theta).all() or not np.isfinite([loss, grad]).all() or grad > 1e-5 or calls <= 0:
        raise ValueError('native head convergence guard failed')
    return theta, loss, grad, calls


def main(outer, source, smoke=False):
    people = [f'P{i:02d}' for i in range(1,11)]
    if outer not in people or source not in people or outer == source: raise ValueError('invalid outer/source people')
    target = ROOT / f'runs/s307/crossfit/{outer}_{source}.npz'
    if target.exists(): raise FileExistsError(target)
    d = protocol.load(); subject = person_ids(d['subject']); y = d['y']; sessions = d['stream']
    fitting = [p for p in people if p not in (outer, source)]
    if len(fitting) != 8: raise ValueError('exactly eight fit people required')
    prior_parts, counts, train_rows, feature_parts, hashes = [], [], [], [], {}
    expected_cache_source = sha(CACHE / 'producer.sha256')
    if expected_cache_source != os.environ['S307_CACHE_LOCK_SHA']:
        raise ValueError('native cache source lock mismatch')
    selected = None
    for person in (p for p in people if p != outer):
        path = CACHE / f'runs/s301_native_inputs/{person}.npz'
        hashes[person] = sha(path)
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata'].item())); rows = z['rows'].copy()
            if (meta['person'] != person or meta['source_sha256'] != expected_cache_source or
                    meta['held_accuracy_computed'] is not False or not np.array_equal(rows, np.flatnonzero(subject == person))):
                raise ValueError('cached person rows or provenance changed')
            if person in fitting:
                prior_parts.append(z['prior'].copy()); counts.append(len(rows))
                train_rows.append(rows); feature_parts.append(z['features'].copy())
            elif person == source:
                selected = {'rows': rows, 'cov': z['cov'].copy(), 'slow': z['slow'].copy()}
    rows = np.concatenate(train_rows); order = np.argsort(rows); rows = rows[order]
    if not np.array_equal(rows, np.flatnonzero(np.isin(subject, fitting))) or np.any(np.isin(subject[rows], [outer,source])):
        raise ValueError('inner source exclusion failed')
    train = np.ascontiguousarray(np.concatenate(feature_parts)[order], dtype=np.float64)
    labels = np.ascontiguousarray(y[rows], dtype=np.int64)
    scaler = StandardScaler().fit(train)
    x = np.ascontiguousarray(scaler.transform(train), dtype=np.float64)
    native = Native(); start = time.monotonic()
    theta, loss, grad, calls = fit_native(native, x, labels)
    prior = np.average(np.stack(prior_parts), axis=0, weights=counts)
    if selected is None or prior.shape != (51,41,41): raise ValueError('missing source/prior')
    payload = {'theta': theta, 'mean': scaler.mean_, 'scale': scaler.scale_, 'fit_rows': rows,
               'prior': np.ascontiguousarray(prior), 'source_rows': selected['rows']}
    with np.load(ROOT / 'runs/shuffle_perms.npz', allow_pickle=False) as z:
        for name in ORDERS[:1] if smoke else ORDERS:
            perm = z[f'{name[:-1]}|{name[-1]}|all_loso'].astype(np.int64)
            if not np.array_equal(np.sort(perm), np.arange(len(subject))): raise ValueError('invalid pinned order')
            ordered = perm[subject[perm] == source]
            values = np.empty((len(selected['rows']), 44239), np.float64)
            for session in dict.fromkeys(sessions[ordered]):
                group = ordered[sessions[ordered] == session]; loc = np.searchsorted(selected['rows'], group)
                values[loc] = native.causal(np.ascontiguousarray(prior), np.ascontiguousarray(selected['cov'][loc]),
                                             np.ascontiguousarray(selected['slow'][loc]))
            if not np.isfinite(values).all(): raise ValueError('nonfinite causal source features')
            payload[f'features|{name}'] = values
            payload[f'order|{name}'] = ordered
    meta = {'outer': outer, 'source': source, 'fit_people': fitting, 'input_sha256': hashes,
            'dataset_sha256': sha(ROOT / 'data/proteus_windows.npz'), 'permutation_sha256': sha(ROOT / 'runs/shuffle_perms.npz'),
            'native_core_sha256': sha(Path(os.environ['NATIVE_CORE_LIB'])), 'loss': loss, 'gradmax': grad,
            'calls': calls, 'fit_and_stream_seconds': time.monotonic() - start, 'held_accuracy_computed': False,
            'smoke': smoke}
    payload['metadata'] = json.dumps(meta, sort_keys=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as f: np.savez(f, **payload)
    print(f'{outer}/{source}: 8-person anchor and source features prepared without scores; {meta["fit_and_stream_seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], len(sys.argv) == 4 and sys.argv[3] == 'smoke')
