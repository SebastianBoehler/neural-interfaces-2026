"""Verify all CPU route sidecars before the existing 90-crossfit verifier."""
import hashlib
import runpy
from pathlib import Path

root = Path.cwd()
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
lock = sha(root/'cpu-src.sha256')
people = [f'P{i:02d}' for i in range(1,11)]
for outer in people:
    for inner in people:
        if outer == inner: continue
        path = root/f'runs/s307/crossfit/{outer}_{inner}.npz'
        recorded = path.with_suffix('.cpu-route').read_text().split()
        if recorded != ['CPU',sha(path),lock]: raise ValueError(f'{outer}/{inner}: CPU route hash mismatch')
runpy.run_path(str(root/'studies/16nov/followups/meta_adaptation/verify_crossfit.py'),run_name='__main__')
