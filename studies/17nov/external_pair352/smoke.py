"""Real PROTEUS-only cost and numerical smoke; never opens target data or accuracy."""
import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np

from data import load_source, sha, source_orders
from geometry import Geometry, digest, fit, predict


def run(root, out):
    d = load_source(root)
    orders = source_orders(root, d)
    x, y = d['X'][:100], d['y'][:100]
    t = time.monotonic()
    bank = Geometry(x, root)
    features_s = time.monotonic() - t
    ids = np.arange(len(x))
    features = bank.training(ids, d['stream'][:100])
    t = time.monotonic()
    models = fit(features, y)
    fit_s = time.monotonic() - t
    assert digest(models) == digest(fit(features, y)), 'Exact training twin differs'
    prior = bank.prior(ids)
    p = predict(models, bank.causal(ids, prior))
    short = predict(models, bank.causal(ids[:17], prior))
    assert all(np.array_equal(p[k][:17], short[k]) for k in models)
    # Same independently computed real-window spectral/covariance path.
    twin = Geometry(x[:7], root)
    assert all(np.array_equal(c[:7], cc) for c, cc in zip(bank.c, twin.c))
    projected = 3 * (features_s * (2655 + 2130) / 100 + 22 * (fit_s + 2))
    result = {'features_100_s': features_s, 'fit_pair_s': fit_s,
              'projected_wall_s': projected, 'maxrss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'real_window_covariance_exact': True, 'training_twin_exact': True,
              'causal_17prefix_exact': True, 'all_8_source_orders_pinned': True,
              'source_data_sha256': sha(root / 'data/proteus_windows.npz'),
              'source_hashes': {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}}
    assert projected < 2400 and result['maxrss_kib'] < 6 * 1024 ** 2
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    run(a.root, a.out)
