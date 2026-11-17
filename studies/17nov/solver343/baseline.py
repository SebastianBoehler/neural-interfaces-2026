"""Matched six-CPU source-only baseline, including cold and ABBA warm fits."""
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


def main():
    lib, x, y, reference, provenance = load()
    if len(os.sched_getaffinity(0)) < 6 or torch.cuda.is_available():
        raise RuntimeError('six-CPU explicit allocation required')
    torch.set_num_threads(6)
    runs = []
    with threadpool_limits(limits=6):
        for phase, order in [('cold', ('sklearn', 'native')), ('warm', ('sklearn', 'native', 'native', 'sklearn'))]:
            for method in order:
                tick, cpu = time.perf_counter(), time.process_time()
                if method == 'sklearn':
                    model = LogisticRegression(C=.3, max_iter=2000, tol=1e-5).fit(x, y)
                    theta = np.column_stack((model.coef_, model.intercept_))
                    evaluations = None
                    iterations = int(model.n_iter_.max())
                else:
                    theta, _, _, evaluations = lib.fit(x, y, False)
                    iterations = None
                wall, core = time.perf_counter()-tick, time.process_time()-cpu
                loss, gradient = lib.objective(x, y, theta, False)
                p = torch.softmax(torch.from_numpy(x) @ torch.from_numpy(theta[:, :-1]).T + torch.from_numpy(theta[:, -1]), 1).numpy()
                row = dict(phase=phase, method=method, wall_s=wall, CPU_core_s=core, objective=loss,
                           max_gradient=float(abs(gradient).max()), evaluations=evaluations, iterations=iterations,
                           process_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                runs.append(row)
                np.savez(Path(os.environ['SOLVER_ROOT']) / f'{phase}_{method}_{len(runs)}.npz', theta=theta, probabilities=p)
                print(json.dumps(row), flush=True)
                Path('progress.json').write_text(json.dumps(runs))
    report = dict(runs=runs, no_held_accuracy=True, matrix_sha256=MATRIX_SHA, library_sha256=LIBRARY_SHA,
                  node=os.uname().nodename, affinity=sorted(os.sched_getaffinity(0)), threadpools=threadpool_info(),
                  source_manifest_sha256=sha('source.sha256'), source_rows_sha256=provenance['rows_sha256'])
    Path('baseline_result.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
