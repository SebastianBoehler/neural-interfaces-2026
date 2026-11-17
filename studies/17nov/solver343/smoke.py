"""Native source-span finite-input, original-objective and full-source guards."""
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from bindings import load, sha, MATRIX_SHA
from span import Span
from verify import objective, parity


def main():
    _, x, y, _, _ = load()
    torch.set_num_threads(6)
    model = Span(Path('build/libspan.so').resolve())
    records = []
    with threadpool_limits(limits=6):
        rows = np.r_[np.flatnonzero(y == 0)[:8], np.flatnonzero(y == 1)[:8], np.flatnonzero(y == 2)[:8]]
        small = x[rows].copy()
        for bad in (np.nan, np.inf, -np.inf):
            fixture = small.copy(); fixture[0, 0] = bad
            try:
                model.fit(fixture, y[rows].copy())
            except RuntimeError as exc:
                if 'nonfinite source' not in str(exc): raise
            else:
                raise ValueError('bad input accepted')
        for name, a, b in [('mini', small, y[rows].copy()), ('full', x, y)]:
            tick = time.perf_counter()
            theta, stats, calls = model.fit(a, b)
            elapsed = time.perf_counter()-tick
            loss, grad, p = objective(a, b, theta)
            if max(abs(loss-stats['objective']), abs(grad-stats['max_gradient'])) > 1e-9 or grad > 1e-5:
                raise ValueError('independent original-space objective/gradient')
            np.savez(f'{name}_fit.npz', theta=theta, probabilities=p)
            row = dict(name=name, seconds=elapsed, stats=stats, calls=calls, independent_loss=loss,
                       independent_max_gradient=grad)
            records.append(row)
            print(json.dumps(row), flush=True)
    Path('smoke_result.json').write_text(json.dumps(dict(passed=True, records=records,
        matrix_sha256=MATRIX_SHA, library_sha256=sha('build/libspan.so'), no_held_accuracy=True), indent=2))


if __name__ == '__main__':
    main()
