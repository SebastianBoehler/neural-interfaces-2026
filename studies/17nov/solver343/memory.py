"""Fresh-process fit memory probes; startup/data loading are outside fit timing."""
import json
import os
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from bindings import load
from span import Span
from reference import FullReference, ReferenceSpan


def resident():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1])
    raise RuntimeError('Linux resident size unavailable')


def worker(method):
    old, x, y, _, _ = load()
    model = Span(Path('build/libspan.so').resolve())
    torch.set_num_threads(6)
    baseline, baseline_hwm = resident(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    samples, stop = [baseline], threading.Event()
    def sample():
        while not stop.wait(.01):
            samples.append(resident())
    thread = threading.Thread(target=sample)
    thread.start()
    with threadpool_limits(limits=6):
        tick = time.perf_counter()
        if method == 'span': model.fit(x, y)
        elif method == 'full': FullReference(Path('build/libspan.so').resolve()).fit(x, y)
        elif method == 'spanref': ReferenceSpan(Path('build/libspan.so').resolve()).fit(x, y)
        elif method == 'native': old.fit(x, y, False)
        elif method == 'sklearn': LogisticRegression(C=.3, max_iter=2000, tol=1e-5).fit(x, y)
        else: raise ValueError(method)
        elapsed = time.perf_counter()-tick
    samples.append(resident()); stop.set(); thread.join()
    result = dict(method=method, wall_s=elapsed, loaded_current_RSS_KiB=baseline,
                  sampled_fit_peak_RSS_KiB=max(samples), sampled_fit_delta_RSS_KiB=max(samples)-baseline,
                  load_process_peak_RSS_KiB=baseline_hwm,
                  process_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  sampling_interval_s=.01, affinity=sorted(os.sched_getaffinity(0)), pid=os.getpid())
    Path(f'memory_{method}.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        worker(sys.argv[1])
    else:
        for method in ('sklearn', 'native', 'span'):
            subprocess.run([sys.executable, '-B', __file__, method], check=True)
        Path('memory_result.json').write_text(json.dumps({k: json.loads(Path(f'memory_{k}.json').read_text())
            for k in ('sklearn', 'native', 'span')}, indent=2))
