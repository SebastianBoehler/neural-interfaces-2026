"""Validate archived control against its actual historical CP source."""
import hashlib
from pathlib import Path

ARCHIVE=Path('/mnt/lustre/work/gehler/gfh527/ni26_lanes292_20261116/lane1/chain')
CP_SHA='49292fce9afbb505d5e39a9a27d4c2686c7140f445d1680fc3085017ee498f73'
ORDERS=('within0','within1','within2','within3','full0','full1','runs0','runs1')
BASES=('B2_pool','B2_s345','B2_s012','B2_s0','B2_s1','B2_s2')


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()


def verify_pair(root,old,name,digest):
 a,b=Path(root)/name,Path(old)/name
 if not a.is_file() or not b.is_file() or sha(a)!=digest or sha(b)!=digest:
  raise ValueError(f'archived control/copy mismatch: {name}')


def install(s,root):
 root=Path(root).resolve(strict=True); archive=ARCHIVE.resolve(strict=True)
 source=archive/'studies/14nov/newmember/s281_cp.py'
 if sha(source)!=CP_SHA:raise ValueError('historical CP source checksum changed')
 lock=root/'runs/coral304_inputs/control-original.sha256'
 records={}
 for line in lock.read_text().splitlines():
  digest,name=line.split()
  if name in records or Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('invalid control manifest')
  records[name]=digest
 required=set()
 for order in ORDERS:
  required.add(f'runs/chainout/s281_v66p_control_{order}.npz')
  required.update(f'runs/newmember/s281_v66p_control_{order}_{x}' for x in ('gate.npz','provenance.json'))
  for base in BASES:
   required.update(f'runs/s281_v66_cp/control_{order}_{base}.{ext}' for ext in ('npz','json'))
 if len(records)!=128 or not required<=set(records):raise ValueError('incomplete control identity manifest')
 for name,digest in records.items():verify_pair(root,archive,name,digest)
 original=s.load_output
 def load_output(actual_root,arm,order,base,n,cache):
  if Path(actual_root).resolve()!=root:raise ValueError('unexpected scorer root')
  if arm!='control':return original(actual_root,arm,order,base,n,cache)
  names=[f'runs/chainout/s281_v66p_control_{order}.npz',
         f'runs/newmember/s281_v66p_control_{order}_gate.npz',
         f'runs/newmember/s281_v66p_control_{order}_provenance.json']
  names += [f'runs/s281_v66_cp/control_{order}_{base}.{ext}' for ext in ('npz','json')]
  for name in names:verify_pair(root,archive,name,records[name])
  return original(archive,arm,order,base,n,cache)
 s.load_output=load_output
 print('all archived controls match pinned copies; historical CP source verified',flush=True)
