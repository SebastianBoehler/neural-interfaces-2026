"""Validate the recorded loss mass independently and give per-class summaries."""
import json
from pathlib import Path
import sys

import numpy as np


def inspect(smoke):
    results = {}
    for arm in ('stock', 'primary', 'placebo'):
        paths = sorted(Path.cwd().glob(f'{arm}_*_smoke.json' if smoke else f'{arm}_[0-8].json'))
        assert len(paths) == (1 if smoke else 9)
        shifts, gpus = [], set()
        for path in paths:
            for stream in json.loads(path.read_text()).values():
                for e in stream['events']:
                    before = np.asarray(e['coefficient_mass_before'])
                    after = np.asarray(e['coefficient_mass_after'])
                    assert np.allclose([before.sum(), after.sum()], 1, atol=1e-12, rtol=0)
                    assert np.array_equal(before == 0, after == 0)
                    assert np.isfinite(before).all() and np.isfinite(after).all()
                    gpus.add(e['gpu'])
                    if e['unequal']:
                        delta = after-before
                        dominant = np.argmax(before)
                        assert {'stock': delta[dominant] == 0, 'primary': delta[dominant] < 0,
                                'placebo': delta[dominant] > 0}[arm]
                        shifts.append((delta, delta[dominant]))
        assert len(shifts) >= 5
        values = np.stack([x[0] for x in shifts])
        results[arm] = dict(unequal_refits=len(shifts), gpu=sorted(gpus),
                mean_shift_per_class=values.mean(0).tolist(),
                mean_dominant_class_shift=float(np.mean([x[1] for x in shifts])))
    Path('smoke_mass.json' if smoke else 'mass.json').write_text(json.dumps(results, indent=2)+'\n')
    print(results, flush=True)


if __name__ == '__main__':
    inspect(sys.argv[1:] == ['smoke'])
