"""Require exact producer/consumer tensors and permutations before chain smoke."""
import hashlib
import json
from pathlib import Path
P=Path('/mnt/lustre/work/gehler/gfh527/ni26_s323_normsplit_20261117')
C=Path('/mnt/lustre/work/gehler/gfh527/ni26_s323_chain_20261117')

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
expected={'data/proteus_windows.npz':'2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327',
          'runs/shuffle_perms.npz':'6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f'}
for name,digest in expected.items():
 if sha(P/name)!=digest or sha(C/name)!=digest:raise ValueError(f'producer/consumer mismatch: {name}')
if sha(P/'source323.sha256')!='88437e69b5eaf7ff84a3635df88fe68b38394abbc527629b0f41c7fcc24eb6d5':raise ValueError('producer source changed')
if sha(C/'representation-source.sha256')!='baf212bddfc79ed4d8bce9a9600c5c3db56d039d72c08f8fcfc92997dafbd55c':raise ValueError('chain source changed')
lock=json.loads((C/'runs/representation_inputs/lock.json').read_text())
if sha(C/'runs/representation_inputs/streams.sha256')!=lock['manifest_sha256']:raise ValueError('stage stream manifest changed')
print('OK: identical pinned producer/consumer data and permutations; source and stream locks',flush=True)
