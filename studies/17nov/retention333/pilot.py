"""One complete fixed nine-bank reconstruction block; never opens labels."""
import json
import os
import time
from pathlib import Path
import numpy as np
from probe import load, mapping, numerical_guard, error_metrics, NAMES, sha, digest


def main():
    root = Path(os.environ['RETENTION_ROOT'])
    guard = json.loads((root / 'guard.json').read_text())
    if not guard['passed'] or not guard['budget_passed'] or guard['source_sha256'] != sha(root / 'source.sha256'):
        raise ValueError('same-source completed guard and measured budget required')
    started = time.monotonic()
    data, banks, manifest = load()
    n = len(data['fit'])
    maps, identities, numerical = {}, {}, {}
    # Every source fit and independent guard finishes before teacher-query errors.
    for name in NAMES:
        tick = time.monotonic()
        bank = banks[name]
        numerical[name] = numerical_guard(bank[:n], bank[n:])
        maps[name], meta = mapping(bank[:n], bank[n:])
        original = maps[name].copy()
        reverse, _ = mapping(bank[:n], bank[n:][::-1])
        query_order_error = float(np.max(np.abs(original - reverse[::-1])))
        if query_order_error > 1e-12 or not np.isfinite(original).all():
            raise ValueError('stateless query-order map guard failed')
        identities[name] = dict(**meta, input_sha256=digest(bank), map_sha256=digest(original),
                                query_order_max_abs=query_order_error, seconds=time.monotonic() - tick)
    with (root / 'fitted_maps.npz').open('xb') as f:
        np.savez(f, source_rows=data['rows'][data['fit']], query_rows=data['rows'][data['query']], **maps)
    fit_features = data['features'][data['fit']]
    with (root / 'target_scaler.npz').open('xb') as f:
        np.savez(f, mean=fit_features.mean(0), sd=fit_features.std(0), source_rows=data['rows'][data['fit']])
    results = {}
    for name in NAMES:
        results[name] = error_metrics(maps[name], fit_features, data['features'][data['query']], data['people'][data['query']])
    report = dict(scope='selected exposed-source label-free linear accessibility only; no classification gain',
                  labels_opened=False, P01_used=False, penalty=.1, source_sha256=sha(root / 'source.sha256'),
                  manifest=manifest['records']['source7'], numerical=numerical, banks=identities, results=results,
                  fitted_maps_sha256=sha(root / 'fitted_maps.npz'), target_scaler_sha256=sha(root / 'target_scaler.npz'),
                  source_rows_sha256=digest(data['rows'][data['fit']]), query_rows_sha256=digest(data['rows'][data['query']]),
                  seconds=time.monotonic() - started)
    with (root / 'results.json').open('x') as f:
        json.dump(report, f, indent=2)
    print('333 complete nine-bank guarded retention block; no labels opened', flush=True)


if __name__ == '__main__':
    main()
