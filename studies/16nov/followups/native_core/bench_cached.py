"""Compare original Python, cached-coefficient Python and native on one CPU."""
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bench_infer import execute, load  # noqa: E402


def main():
    path = Path(os.environ['S301_CACHED_OUTPUT'])
    if path.exists():
        raise FileExistsError(path)
    smoke = json.loads(Path(os.environ['S301_CACHED_SMOKE']).read_text())
    if (smoke['held_accuracy_computed'] is not False or
            smoke['source_sha256'] != hashlib.sha256((Path(os.environ['NI26_WS']) / 'cached.sha256').read_bytes()).hexdigest() or
            smoke['reference_sha256'] != hashlib.sha256(Path(os.environ['S301_BENCH_REFERENCE']).read_bytes()).hexdigest()):
        raise ValueError('cached-coefficient smoke gate differs')
    inputs = load()
    orders = (('python', 'python_cachedcoeff', 'native'),
              ('native', 'python_cachedcoeff', 'python'),
              ('python_cachedcoeff', 'python', 'native'))
    runs = []
    for order in orders:
        for method in order:
            runs.append(execute(method, 0, inputs))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump({'runs': runs, 'held_accuracy_computed': False,
                   'reference_sha256': hashlib.sha256(Path(os.environ['S301_BENCH_REFERENCE']).read_bytes()).hexdigest(),
                   'benchmark_source_sha256': hashlib.sha256((Path(os.environ['NI26_WS']) / 'cached.sha256').read_bytes()).hexdigest(),
                   'interpretation': 'same CPU/raw stream/scaler/head; native batched eigensolvers and Python loops remain distinct implementations'}, f, indent=2, sort_keys=True)
        f.write('\n')
    print(f'cached-coefficient paired comparison complete: {path}; no score', flush=True)


if __name__ == '__main__':
    main()
