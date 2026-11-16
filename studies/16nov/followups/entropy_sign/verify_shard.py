"""Require an unchanged verified §305 source shard for one §310 person."""
import json
import os
import sys
from pathlib import Path
from hashlib import sha256

held=sys.argv[1]
root=Path(os.environ['S310_S305']).resolve(strict=True)
assembler=Path(os.environ['S310_S305_ASSEMBLER']).resolve(strict=True)
path=root/f'runs/s305_converged_shards/{held}.npz'
h=sha256()
with path.open('rb') as f:
 for b in iter(lambda:f.read(1<<20),b''):h.update(b)
lock=json.loads((assembler/'runs/s305_converged/manifest.json').read_text())
if lock['shards'].get(f'{held}.npz')!=h.hexdigest(): raise ValueError('§305 verified shard mismatch')
print(held,'§305 shard matched full manifest; no score',flush=True)
