"""Matched whole-fit/time/memory and two-fold qualification of the full-D repair."""
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_info, threadpool_limits
from bindings import load, sha, MATRIX_SHA
from reference import FullReference
from sklearn_reference import fit as sklearn_fit
from held import compare
from second_fold import load as load_p10
from verify import objective, parity


def fit(method, x, y):
    tick, cpu = time.perf_counter(), time.process_time()
    if method == 'full':
        theta, stats, _ = FullReference(Path('build/libspan.so').resolve()).fit(x, y)
    else:
        model, stats = sklearn_fit(x, y)
        theta = np.column_stack((model.coef_, model.intercept_))
    wall, core = time.perf_counter()-tick, time.process_time()-cpu
    loss, gradient, p = objective(x, y, theta)
    return theta, p, dict(method=method, wall_s=wall, CPU_core_s=core, objective=loss,
                         max_gradient=gradient, optimizer=stats,
                         cumulative_process_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def paired(held, x, y, phases):
    records, reference_theta, reference_p, target = [], None, None, None
    for phase, order in phases:
        for method in order:
            theta, p, row = fit(method, x, y)
            if reference_theta is None:
                if method != 'sklearn': raise ValueError('reference must run first')
                reference_theta, reference_p, target = theta, p, row['objective']+1e-5
            source = parity(p, reference_p)
            held_parity = compare(held, theta, reference_theta)
            passed = (row['objective'] <= target and row['max_gradient'] <= 1e-5
                      and source['max_probability_error'] <= 1e-3 and source['non_tied_changes'] == 0
                      and all(r['passed'] for r in held_parity.values()))
            row.update(phase=phase, held=held, quality_passed=bool(passed), source_parity=source, held_parity=held_parity)
            records.append(row)
            print(json.dumps(row), flush=True)
            Path(f'{held}_reference_progress.json').write_text(json.dumps(records, indent=2))
    return records


def main():
    proof = json.loads(Path('reference_smoke_result.json').read_text())
    if not proof['records']['full']['passed'] or proof['library_sha256'] != sha('build/libspan.so'):
        raise ValueError('full-D source/held smoke must pass unchanged before benchmark')
    _, x, y, _, source = load()
    torch.set_num_threads(6)
    with threadpool_limits(limits=6):
        p01 = paired('P01', x, y, [('cold', ('sklearn', 'full')),
                                 ('warm', ('sklearn', 'full', 'full', 'sklearn'))])
        old = Path('/mnt/lustre/work/gehler/gfh527')
        candidate = np.load(old/'ni26_solver343_span_v1_20261117/full_fit.npz')['theta']
        native = np.load(old/'ni26_solver343_baseline_v1_20261117/cold_native_2.npz')['theta']
        diagnosis = compare('P01', candidate, native)
    del x, y
    memory = {}
    for method in ('sklearn', 'full'):
        subprocess.run([sys.executable, '-B', 'memory.py', method], check=True)
        memory[method] = json.loads(Path(f'memory_{method}.json').read_text())
    x, y, p10_provenance = load_p10()
    with threadpool_limits(limits=6):
        p10 = paired('P10', x, y, [('cold', ('sklearn', 'full'))])
    report = dict(passed=all(r['quality_passed'] for r in p01+p10), P01=p01, P10=p10,
                  memory=memory, old_native_vs_span_diagnosis=diagnosis, p10_provenance=p10_provenance,
                  matrix_sha256=MATRIX_SHA, source_rows_sha256=source['rows_sha256'],
                  library_sha256=sha('build/libspan.so'), source_manifest_sha256=sha('source.sha256'),
                  no_held_accuracy=True, node=os.uname().nodename,
                  affinity=sorted(os.sched_getaffinity(0)), threadpools=threadpool_info(),
                  timing_boundary='validation+customC++valuegradient+sameSciPyLBFGSB+fullWoutput; independentpostfitverificationexcludedboth',
                  scope='full-width single-validation C++analytical objective and sklearn-reference optimizer; noGram/factor/devicechange')
    Path('reference_benchmark_result.json').write_text(json.dumps(report, indent=2))
    if not report['passed']:
        raise ValueError('strict full-D two-fold qualification failed')


if __name__ == '__main__':
    main()
