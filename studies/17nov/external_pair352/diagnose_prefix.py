"""Source-only diagnosis of the preserved bitwise prefix failure."""
import argparse
import json
from pathlib import Path

import numpy as np

from data import load_source
from geometry import Geometry, fit, predict

p = argparse.ArgumentParser(__doc__)
p.add_argument('--root', type=Path, required=True)
a = p.parse_args()
d = load_source(a.root)
b = Geometry(d['X'][:100], a.root)
i = np.arange(100)
m = fit(b.training(i, d['stream'][:100]), d['y'][:100])
prior = b.prior(i)
f, g = b.causal(i, prior), b.causal(i[:17], prior)
x, y = predict(m, f), predict(m, g)
r = {}
for arm in m:
    scaler, head = m[arm]
    rowwise = lambda z: np.concatenate([head.predict_proba(scaler.transform(row[None])) for row in z])
    r[arm] = {'feature_bitwise': np.array_equal(f[arm][:17], g[arm]),
              'probability_max_abs': float(np.max(np.abs(x[arm][:17] - y[arm]))),
              'changed_labels': int((x[arm][:17].argmax(1) != y[arm].argmax(1)).sum()),
              'rowwise_probability_bitwise': np.array_equal(rowwise(f[arm])[:17], rowwise(g[arm]))}
print(json.dumps(r))
