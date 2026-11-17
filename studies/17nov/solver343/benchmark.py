"""Source-only complete fits on one node, with all factorization costs included."""
import json
import os
import resource
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_info, threadpool_limits
from bindings import load, sha, MATRIX_SHA, LIBRARY_SHA
from span import Span
from verify import objective, parity


def main():
    root = Path(os.environ['SOLVER_ROOT'])
    if (root/'result.json').exists():
        raise FileExistsError(root/'result.json')
    proof = json.loads((root/'smoke_result.json').read_text())
    if not proof['passed'] or proof['matrix_sha256'] != MATRIX_SHA or proof['library_sha256'] != sha('build/libspan.so'):
        raise ValueError('smoke identity changed')
    lib, x, y, _, source = load()
    span = Span(root/'build/libspan.so')
    if len(os.sched_getaffinity(0)) < 6 or torch.cuda.is_available():
        raise RuntimeError('explicit sixCPU noGPU route required')
    torch.set_num_threads(6)
    runs, reference, target = [], None, None
    with threadpool_limits(limits=6):
        schedule = [('cold', ('sklearn', 'span', 'native')),
                    ('warm', ('sklearn', 'span', 'native', 'native', 'span', 'sklearn'))]
        for phase, order in schedule:
            for method in order:
                tick, cpu = time.perf_counter(), time.process_time()
                if method == 'span':
                    theta, stages, calls = span.fit(x, y)
                    iterations = None
                elif method == 'native':
                    theta, _, _, calls = lib.fit(x, y, False)
                    stages, iterations = {}, None
                else:
                    model = LogisticRegression(C=.3, max_iter=2000, tol=1e-5).fit(x, y)
                    theta = np.column_stack((model.coef_, model.intercept_))
                    iterations, calls, stages = int(model.n_iter_.max()), None, {}
                elapsed, core = time.perf_counter()-tick, time.process_time()-cpu
                loss, grad, probabilities = objective(x, y, theta)
                if reference is None:
                    reference, target = probabilities.copy(), loss+1e-5
                identity = parity(probabilities, reference)
                passed = loss <= target and grad <= 1e-5 and identity['max_probability_error'] <= 1e-3 and identity['non_tied_changes'] == 0
                row = dict(phase=phase, method=method, wall_s=elapsed, CPU_core_s=core,
                           objective=loss, max_gradient=grad, evaluations=calls, iterations=iterations,
                           stages=stages, parity_to_cold_sklearn=identity, quality_passed=bool(passed),
                           process_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                runs.append(row)
                np.savez(root/f'fit_{len(runs)}.npz', theta=theta)
                (root/'progress.json').write_text(json.dumps(runs, indent=2))
                print(json.dumps(row), flush=True)
    report = dict(runs=runs, objective_target=target, quality_passed=all(r['quality_passed'] for r in runs),
                  no_held_accuracy=True, matrix_sha256=MATRIX_SHA, old_library_sha256=LIBRARY_SHA,
                  source_rows_sha256=source['rows_sha256'], library_sha256=sha('build/libspan.so'),
                  source_manifest_sha256=sha('source.sha256'), node=os.uname().nodename,
                  torch_version=torch.__version__, torch_config=torch.__config__.show(),
                  threadpools=threadpool_info(), affinity=sorted(os.sched_getaffinity(0)),
                  timing_boundary='fitAPI: validation,Gram,eigh,reducedsolve,fullWreconstruction,fullspacefinalcheck; baselineAPI normalfinalcheck',
                  target_note='fulltermination upper bound only; no accepted-iterate common-quality timer')
    (root/'result.json').write_text(json.dumps(report, indent=2))
    if not report['quality_passed']:
        raise ValueError('preregistered quality/prediction guard failed; inspect preserved report')


if __name__ == '__main__':
    main()
