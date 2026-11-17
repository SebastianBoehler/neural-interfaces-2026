"""Label-free linear retention of all source-fixed covariance/slow coordinates."""
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve

FIXED = Path('/mnt/lustre/work/gehler/gfh527/ni26_gen334_v4_20261117/fixed_features')
EMBED = Path('/mnt/lustre/work/gehler/gfh527/ni26_labram329_A100source_v1_20261117/source_banks.npz')
MANIFEST_SHA = 'c8ebeac134d610dc78111700ae84f5e61d74fdf0f500510ba079fca2fdc7bf9e'
FEATURE_SHA = 'abb931da647bc1613e987ef04e3bc0f1d7c99abece389cf6d3fe58180d15eb09'
EMBED_SHA = '524610d6f21633a2e93cda04007dc82996215ef60a0f14a0bbd28edd9ca2541a'
NAMES = ('raw287', 'pretrained_pooled', 'random_pooled', 'pretrained_retained',
         'random_retained', 'pretrained_pooled_raw', 'random_pooled_raw',
         'pretrained_retained_raw', 'random_retained_raw')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def load():
    if sha(FIXED / 'manifest.json') != MANIFEST_SHA or sha(FIXED / 'source7.npz') != FEATURE_SHA or sha(EMBED) != EMBED_SHA:
        raise ValueError('immutable input identity failed')
    manifest = json.loads((FIXED / 'manifest.json').read_text())
    if manifest['labels_opened'] or manifest['P01_used']:
        raise ValueError('teacher source boundary failed')
    with np.load(FIXED / 'source7.npz', allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    fit, query = data['fit'], data['query']
    if not np.array_equal(fit, data['source']) or set(data['people'][fit]) != {f'P{i:02d}' for i in range(4, 11)} or set(data['people'][query]) != {'P02', 'P03'}:
        raise ValueError('source7/query partition changed')
    if 'P01' in data['people'] or len(np.intersect1d(fit, query)):
        raise ValueError('excluded-person overlap')
    record = manifest['records']['source7']
    if digest(data['rows'][fit]) != record['source_rows_sha256'] or digest(data['reference']) != record['reference_sha256'] or digest(data['slow_mean']) != record['slowmean_sha256']:
        raise ValueError('reference/row/slow identity failed')
    with np.load(EMBED, allow_pickle=False) as z:
        if set(z.files) != set(NAMES) | {'source_rows', 'query_rows'}:
            raise ValueError('complete nine frozen banks required')
        if not np.array_equal(z['source_rows'], data['rows'][fit]) or not np.array_equal(z['query_rows'], data['rows'][query]):
            raise ValueError('teacher/embedding row order differs')
        banks = {name: z[name].astype(np.float64) for name in NAMES}
    n = len(fit) + len(query)
    if data['features'].shape != (n, 44239) or any(v.shape[0] != n or not np.isfinite(v).all() for v in banks.values()):
        raise ValueError('frozen feature layout/finite guard')
    return data, banks, manifest


def mapping(source, query, penalty=.1):
    """Return the exact source-row weights for centered ridge predictions."""
    n = len(source)
    mean, sd = source.mean(0), source.std(0)
    if not np.isfinite(sd).all():
        raise ValueError('nonfinite source scale')
    sd = np.where(sd > 1e-12, sd, 1.)
    x, q = (source - mean) / sd, (query - mean) / sd
    factor = cho_factor(x.T @ x / n + penalty * np.eye(x.shape[1]), lower=True)
    weights = cho_solve(factor, q.T).T @ x.T / n
    return weights, dict(width=x.shape[1], constant_coordinates=int(np.sum(source.std(0) <= 1e-12)),
                         mean_sha256=digest(mean), scale_sha256=digest(sd))


def numerical_guard(source, query):
    # Real frozen coordinates; the independent augmented least-squares oracle
    # does not share Cholesky or the source-row prediction implementation.
    x, q = source[:24, :12], query[:4, :12]
    y = source[:24, 12:20]
    mu, sd = x.mean(0), x.std(0)
    xx, qq = (x - mu) / sd, (q - mu) / sd
    yc = y - y.mean(0)
    augmented = np.r_[xx, np.sqrt(len(x) * .1) * np.eye(12)]
    target = np.r_[yc, np.zeros((12, y.shape[1]))]
    coef = np.linalg.lstsq(augmented, target, rcond=None)[0]
    weights, _ = mapping(x, q)
    error = float(np.max(np.abs(weights @ yc - qq @ coef)))
    reverse, _ = mapping(x[::-1], q)
    order_error = float(np.max(np.abs(weights @ yc - reverse @ yc[::-1])))
    if error > 1e-9 or order_error > 1e-9:
        raise ValueError(f'independent ridge/order oracle failed: {error}, {order_error}')
    return dict(least_squares_max_abs=error, source_order_max_abs=order_error)


def groups():
    i, j = np.triu_indices(41)
    diagonal = np.concatenate([b * 861 + np.flatnonzero(i == j) for b in range(51)])
    offdiagonal = np.concatenate([b * 861 + np.flatnonzero(i != j) for b in range(51)])
    return dict(diagonal=diagonal, offdiagonal=offdiagonal, slow=np.arange(43911, 44239))


def error_metrics(weights, yfit, yquery, people):
    buckets = groups()
    total = {p: {k: [0., 0] for k in buckets} for p in ('P02', 'P03')}
    baseline = {p: {k: [0., 0] for k in buckets} for p in ('P02', 'P03')}
    constants = 0
    for start in range(0, 44239, 1024):
        stop = min(start + 1024, 44239)
        train, held = yfit[:, start:stop], yquery[:, start:stop]
        mean, sd = train.mean(0), train.std(0)
        active = sd > 1e-12
        constants += int((~active).sum())
        scale = np.where(active, sd, 1.)
        a, b = (train - mean) / scale, (held - mean) / scale
        pred = weights @ a
        for person in total:
            rows = people == person
            for name, columns in buckets.items():
                local = columns[(columns >= start) & (columns < stop)] - start
                local = local[active[local]]
                total[person][name][0] += float(np.square(pred[rows][:, local] - b[rows][:, local]).sum())
                total[person][name][1] += int(rows.sum() * len(local))
                baseline[person][name][0] += float(np.square(b[rows][:, local]).sum())
                baseline[person][name][1] += int(rows.sum() * len(local))
    out = {}
    for person in total:
        out[person] = {k: dict(source_standardized_MSE=v[0] / v[1],
                              source_mean_MSE=baseline[person][k][0] / v[1],
                              relative_to_source_mean=v[0] / baseline[person][k][0],
                              row_coordinates=v[1]) for k, v in total[person].items()}
    return dict(people=out, constant_target_coordinates=constants)


def main():
    started = time.monotonic()
    root = Path(os.environ['RETENTION_ROOT'])
    data, banks, manifest = load()
    n = len(data['fit'])
    proof = numerical_guard(banks['raw287'][:n], banks['raw287'][n:])
    # Cost is measured on source rows only, before held reconstruction errors.
    largest = banks['pretrained_retained_raw']
    tick = time.monotonic()
    weights, meta = mapping(largest[:n], largest[:len(data['query'])])
    map_seconds = time.monotonic() - tick
    teacher = data['features'][data['fit']]
    tick = time.monotonic()
    for _ in range(3):
        weights @ teacher[:, :1024]
    block_seconds = (time.monotonic() - tick) / 3
    # Probe uses the full query row count, but ONLY training inputs/targets.
    projection = 2 * len(NAMES) * (map_seconds + block_seconds * 44) + 60
    guard = dict(passed=True, numerical=proof, source_only_cost_seconds=dict(mapping=map_seconds, target1024=block_seconds),
                 projected_seconds=projection, budget_passed=projection <= 1200,
                 feature_sha256=FEATURE_SHA, embeddings_sha256=EMBED_SHA, source_sha256=sha(root / 'source.sha256'),
                 labels_opened=False, P01_used=False)
    with (root / 'guard.json').open('x') as f:
        json.dump(guard, f, indent=2)
    if not guard['budget_passed']:
        raise ValueError(f'cost gate stops before reconstruction results: {projection:.1f}s >1200s')
    print('333 source-only guard complete', json.dumps(guard), flush=True)
    if os.environ['RETENTION_PHASE'] == 'guard':
        return
    raise ValueError('pilot requires separately released guarded entry point')


if __name__ == '__main__':
    main()
