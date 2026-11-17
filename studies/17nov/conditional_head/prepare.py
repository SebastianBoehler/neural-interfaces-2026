"""Build seven-person-only strong-core features and causal source-query streams."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
BASE = Path('/mnt/lustre/work/gehler/gfh527')
BRIDGE = BASE/'ni26_s307_meta_20261116/cpu_smoke_v2/studies/16nov/followups/native_core/bridge.py'
if hashlib.sha256(BRIDGE.read_bytes()).hexdigest() != '78e70fa1c4a4e43206b96b2a12d2df4212ed6f9b2c4f2827a683f8cb48d7b047':
    raise ValueError('changed native bridge')
sys.path.insert(0, str(BRIDGE.parent))
from bridge import Native, pointer

CACHE = BASE / 'ni26_s301_native_producer_v3_20261116'
OLD = BASE / 'ni26_s307_meta_20261116/cpu_smoke_v2'
FIT = tuple(f'P{i:02d}' for i in range(4, 11))
QUERY = ('P02', 'P03')
ORDERS = ('within0', 'within1', 'within2', 'within3', 'full0', 'full1', 'runs0', 'runs1')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def causal(native, prior, cov, slow):
    handle = native.core.nc_create(pointer(prior), 0)
    if not handle: raise RuntimeError(native.core.nc_error().decode())
    out = np.empty((len(cov), 44239), np.float64)
    try:
        for i in range(len(cov)):
            native.check(native.core.nc_step(handle, pointer(cov[i]), pointer(slow[i]), pointer(out[i])), 'nc_error')
    finally:
        native.core.nc_destroy(handle)
    return out


def main():
    start = time.monotonic()
    target = Path('runs/s326')
    target.mkdir(parents=True, exist_ok=False)
    data_path = OLD / 'data/proteus_windows.npz'
    perm_path = OLD / 'runs/shuffle_perms.npz'
    native_path = Path(os.environ['NATIVE_CORE_LIB'])
    train_path = Path(os.environ['NATIVE_TRAIN_LIB'])
    expected = {data_path:'2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327',
                perm_path:'6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f',
                native_path:'4bcbb983086f690780c5690b25fcf2f429a88ff3377d108a7f4b12535fa0fb56',
                train_path:'1508f45481b8a396d181b99e07cd72206383e51b8674e876cacadc5dbae131de'}
    for path, digest in expected.items():
        if sha(path) != digest: raise ValueError(f'changed pinned input {path}')
    producer_sha = sha(CACHE / 'producer.sha256')
    if producer_sha != '21c1af2e0784329da2f662319ecdac3f6757715cbcece604fa1db46d21b659bf':
        raise ValueError('changed native cache producer')
    lock = {name:digest for digest,name in (line.split() for line in (OLD/'cache.sha256').read_text().splitlines())}
    with np.load(data_path, allow_pickle=False) as z:
        subject = np.asarray([str(v).rsplit('/', 1)[-1] for v in z['subject']])
        session = np.char.add(np.char.add(subject, '|'), z['session'])
        labels = z['y'].copy()
    parts, indices, priors, counts, selected, hashes = [], [], [], [], {}, {}
    for person in FIT + QUERY:
        name = f'runs/s301_native_inputs/{person}.npz'; path = CACHE/name
        digest = sha(path); hashes[person] = digest
        if digest != lock[name]: raise ValueError(f'changed cache {person}')
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata'].item())); rows = z['rows'].copy()
            if (meta['person'] != person or meta['source_sha256'] != producer_sha or
                meta['held_accuracy_computed'] is not False or not np.array_equal(rows, np.flatnonzero(subject == person))):
                raise ValueError(f'cache provenance/rows failed {person}')
            if person in FIT:
                parts.append(z['features'].copy()); indices.append(rows)
                priors.append(z['prior'].copy()); counts.append(len(rows))
            else:
                selected[person] = (rows, z['cov'].copy(), z['slow'].copy())
    idx = np.concatenate(indices); order = np.argsort(idx); idx = idx[order]
    if not np.array_equal(idx, np.flatnonzero(np.isin(subject, FIT))) or np.isin(subject[idx], ('P01',)+QUERY).any():
        raise ValueError('seven-person source exclusion failed')
    raw = np.concatenate(parts)[order]; scaler = StandardScaler().fit(raw)
    x = np.ascontiguousarray(scaler.transform(raw), dtype=np.float64)
    y = np.ascontiguousarray(labels[idx], dtype=np.int64)
    groups = np.asarray(subject[idx]); delta = []
    for cls in range(3):
        means = []
        for person in FIT:
            rows = (groups == person) & (y == cls)
            if not rows.any(): raise ValueError(f'missing source stratum {person}/{cls}')
            means.append(x[rows].mean(0))
        means = np.stack(means); delta.extend(means - means.mean(0))
    delta = np.ascontiguousarray(delta, dtype=np.float64)
    if delta.shape != (21, 44239) or not np.isfinite(delta).all(): raise ValueError('bad D')
    prior = np.ascontiguousarray(np.average(np.stack(priors), axis=0, weights=counts))
    with (target/'train.npz').open('xb') as f:
        np.savez(f, x=x, y=y, rows=idx, D=delta, prior=prior, mean=scaler.mean_, scale=scaler.scale_)
    native = Native(); stream_hashes = {}; reset_checks = 0
    with np.load(perm_path, allow_pickle=False) as z:
        for name in ORDERS:
            perm = z[f'{name[:-1]}|{name[-1]}|all_loso'].astype(np.int64)
            if not np.array_equal(np.sort(perm), np.arange(len(subject))): raise ValueError('bad permutation')
            for person in QUERY:
                rows, cov, slow = selected[person]; delivered = perm[subject[perm] == person]
                values = np.empty((len(rows), 44239), dtype=np.float64)
                for sess in dict.fromkeys(session[delivered]):
                    group = delivered[session[delivered] == sess]; loc = np.searchsorted(rows, group)
                    values[loc] = causal(native, prior, np.ascontiguousarray(cov[loc]), np.ascontiguousarray(slow[loc]))
                    n = min(61, len(loc))
                    again = causal(native, prior, np.ascontiguousarray(cov[loc[:n]]), np.ascontiguousarray(slow[loc[:n]]))
                    np.testing.assert_array_equal(again, values[loc[:n]])
                    reset_checks += 1
                if not np.isfinite(values).all(): raise ValueError('nonfinite query features')
                path = target/f'{person}_{name}.npz'
                with path.open('xb') as f: np.savez(f, rows=rows, features=values)
                stream_hashes[path.name] = sha(path)
    info = {'fit_people': FIT, 'queries': QUERY, 'source_rows':len(idx), 'cache_sha256':hashes,
            'data_sha256':expected[data_path], 'permutation_sha256':expected[perm_path],
            'producer_sha256':producer_sha, 'core_sha256':expected[native_path], 'train_library_sha256':expected[train_path],
            'train_sha256':sha(target/'train.npz'), 'streams_sha256':stream_hashes, 'reset_checks':reset_checks,
            'held_accuracy_computed':False, 'elapsed_s':time.monotonic()-start}
    with (target/'prepare.json').open('x') as f: json.dump(info, f, indent=2, sort_keys=True)
    print(json.dumps(info, sort_keys=True), flush=True)


if __name__ == '__main__': main()
