"""Validate original/cached/native CPU paths on the same real prefix."""
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_infer import execute, load  # noqa: E402

target = Path(os.environ['S301_CACHED_SMOKE'])
if target.exists():
    raise FileExistsError(target)
inputs = load()
for method in ('python', 'python_cachedcoeff', 'native'):
    execute(method, 0, inputs, limit=6)
target.parent.mkdir(parents=True, exist_ok=True)
with target.open('x') as f:
    json.dump({'source_sha256': hashlib.sha256((Path(os.environ['NI26_WS']) / 'cached.sha256').read_bytes()).hexdigest(),
               'reference_sha256': hashlib.sha256(Path(os.environ['S301_BENCH_REFERENCE']).read_bytes()).hexdigest(),
               'held_accuracy_computed': False}, f, sort_keys=True)
    f.write('\n')
print(f'original/cached/native CPU real-window smoke passed: {target}; no score', flush=True)
