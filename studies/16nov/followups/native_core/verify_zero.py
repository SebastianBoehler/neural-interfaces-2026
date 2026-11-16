"""Reuse §295 exact chain/gate/CP verification for the native zero arm."""
import os
import sys
from pathlib import Path

root = Path(os.environ['NI26_WS']).resolve(strict=True)
path = root / 'studies/16nov/followups/geometry/verify_zero.py'
source = path.read_text()
if source.count('geo_zero_') != 3:
    raise ValueError('unexpected existing zero verifier')
source = source.replace('geo_zero_', 'Rzero_')
exec(compile(source, str(path), 'exec'), {'__name__': '__main__', '__file__': str(path), 'os': os, 'sys': sys})
