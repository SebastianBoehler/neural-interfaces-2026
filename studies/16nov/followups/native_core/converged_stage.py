"""Use guarded native staging with the complete §305 matrix."""
import os
import sys
from pathlib import Path

path = Path(__file__).with_name('stage.py')
source = path.read_text()
replacements = {
    'from member_chain import ARMS, ORDERS, validate_delta': 'from converged_chain import ARMS, ORDERS\nfrom member_chain import validate_delta',
    "source / 'producer.sha256'": "source / 'converged.sha256'",
    "source / 'runs/s301_native/manifest.json'": "source / 'runs/s305_converged/manifest.json'",
    "source / 'runs/s301_native_shards'": "source / 'runs/s305_converged_shards'",
    "source / 'runs/s301_native/deltas'": "source / 'runs/s305_converged/deltas'",
}
for old, new in replacements.items():
    if source.count(old) != 1:
        raise ValueError(f'unexpected native staging anchor: {old}')
    source = source.replace(old, new, 1)
exec(compile(source, str(path), 'exec'), {'__name__': '__main__', '__file__': str(path), 'os': os, 'sys': sys})
