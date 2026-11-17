"""Independent P10 source fold: exact training-row exclusion, no held labels."""
import json
import os
import resource
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from bindings import sha
from held import compare
from span import Span
from verify import objective, parity

BASE = Path('/mnt/lustre/work/gehler/gfh527')
PRODUCER = BASE/'ni26_s301_native_producer_v3_20261116'
REFERENCES = BASE/'ni26_s301_native_reference_all_20261116/runs/s301_reference_all'


def load():
    manifest = json.loads((REFERENCES/'manifest.json').read_text())
    refpath = REFERENCES/'P10.npz'
    if sha(refpath) != manifest['files']['P10.npz'] or manifest['held_accuracy_computed']:
        raise ValueError('source-only reference identity')
    with np.load(refpath, allow_pickle=False) as z:
        rows, mean, scale = z['fit_rows'].copy(), z['mean'].copy(), z['scale'].copy()
    with np.load(PRODUCER/'data/proteus_windows.npz', allow_pickle=False) as z:
        subjects = z['subject']
        expected = np.flatnonzero(subjects != 'Dreyer2026Proteus/P10')
        if not np.array_equal(rows, expected):
            raise ValueError('P10 held exclusion differs')
        y = np.ascontiguousarray(z['y'][rows], dtype=np.int64)
    x, seen = np.empty((len(rows), 44239), np.float64), np.zeros(len(rows), bool)
    pins = {}
    for person in range(1, 10):
        path = PRODUCER/f'runs/s301_native_inputs/P{person:02d}.npz'
        pins[path.name] = sha(path)
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata'].item()))
            indices = z['rows'].copy()
            if meta['person'] != f'P{person:02d}' or meta['held_accuracy_computed'] or meta['source_sha256'] != sha(PRODUCER/'producer.sha256'):
                raise ValueError('cached source provenance')
            positions = np.searchsorted(rows, indices)
            if np.any(positions >= len(rows)) or not np.array_equal(rows[positions], indices) or seen[positions].any():
                raise ValueError('source row coverage differs')
            x[positions] = z['features']
            seen[positions] = True
    if not seen.all() or not np.isfinite(x).all():
        raise ValueError('incomplete source matrix')
    x -= mean
    x /= scale
    return x, y, dict(input_sha256=pins, reference_sha256=sha(refpath),
                     source_rows_sha256=__import__('hashlib').sha256(rows.tobytes()).hexdigest(), held_labels_used=False)


def main():
    x, y, provenance = load()
    torch.set_num_threads(6)
    span = Span(Path('build/libspan.so').resolve())
    fits, parameters, probabilities = [], {}, {}
    with threadpool_limits(limits=6):
        for method in ('sklearn', 'span'):
            tick, cpu = time.perf_counter(), time.process_time()
            if method == 'span':
                theta, stages, evaluations = span.fit(x, y)
                iterations = None
            else:
                model = LogisticRegression(C=.3, max_iter=2000, tol=1e-5).fit(x, y)
                theta = np.column_stack((model.coef_, model.intercept_))
                stages, evaluations, iterations = {}, None, int(model.n_iter_.max())
            elapsed, core = time.perf_counter()-tick, time.process_time()-cpu
            loss, grad, p = objective(x, y, theta)
            parameters[method], probabilities[method] = theta, p
            row = dict(method=method, wall_s=elapsed, CPU_core_s=core, objective=loss, max_gradient=grad,
                       stages=stages, evaluations=evaluations, iterations=iterations)
            fits.append(row); print(json.dumps(row), flush=True)
        source_parity = parity(probabilities['span'], probabilities['sklearn'])
        held = compare('P10', parameters['span'], parameters['sklearn'])
    passed = (fits[1]['objective'] <= fits[0]['objective']+1e-5 and fits[1]['max_gradient'] <= 1e-5
              and source_parity['max_probability_error'] <= 1e-3 and source_parity['non_tied_changes'] == 0
              and all(row['passed'] for row in held.values()))
    report = dict(passed=passed, fits=fits, source_parity=source_parity, held_parity=held,
                  provenance=provenance, source_shape=list(x.shape), no_held_accuracy=True,
                  library_sha256=sha('build/libspan.so'), node=os.uname().nodename,
                  affinity=sorted(os.sched_getaffinity(0)),
                  process_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    Path('second_fold_result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(passed=passed, source_parity=source_parity, held_parity=held)), flush=True)
    if not passed:
        raise ValueError('independent-fold prediction/quality qualification failed')


if __name__ == '__main__':
    main()
