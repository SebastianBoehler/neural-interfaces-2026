"""No-score real-window smoke before the independent performance benchmark."""
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bench_infer import cached, execute, load  # noqa: E402

output = Path(os.environ['S301_BENCH_SMOKE'])
if output.exists():
    raise FileExistsError(output)
inputs = load()
for mode, device in (('python', 0), ('native', 0), ('native', 1)):
    execute(mode, device, inputs, limit=6)
for device in (0, 1):
    cached(device, inputs)
digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
output.parent.mkdir(parents=True, exist_ok=True)
with output.open('x') as f:
    json.dump({'benchmark_sha256': digest(Path(os.environ['NI26_WS']) / 'benchmark.sha256'),
               'reference_sha256': digest(os.environ['S301_BENCH_REFERENCE']),
               'native_core_sha256': digest(os.environ['NATIVE_CORE_LIB']),
               'held_accuracy_computed': False}, f, sort_keys=True)
    f.write('\n')
print(f'benchmark real-window CPU/GPU and cache smoke passed: {output}; no score', flush=True)
