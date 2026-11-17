"""Isolate the complete branches' owning imports from the waveform stack."""
from pathlib import Path
import hashlib
import sys
import numpy as np

root = Path('/mnt/lustre/work/gehler/gfh527')
owner = root / 'ni26_ceiling362_v1_20261117'
manifest = owner / 'source.sha256'
assert hashlib.sha256(manifest.read_bytes()).hexdigest() == '4ae0973593df4ee4f2482fd717f590c52e13c4e7956e73b29f92abe5c751b6f9'
for line in manifest.read_text().splitlines():
    digest, name = line.split(maxsplit=1)
    assert hashlib.sha256((owner / name).read_bytes()).hexdigest() == digest
sys.path.insert(0, str(owner))
from oracle import CHAIN, validate

with np.load(CHAIN/'data/proteus_windows.npz') as z:
    meta = {key: z[key] for key in ('subject', 'session', 'interface')}
scorer, pairs, people = validate(meta)
out = {'subject': meta['subject']}
for j, arm in enumerate(('frozen', 'JS')):
    for i, order in enumerate(scorer.ORDERS):
        out[arm+'_'+order] = pairs[j, i, 0]
np.savez_compressed('controls.npz', **out)
