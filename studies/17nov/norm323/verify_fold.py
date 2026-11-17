"""Check one complete producer fold without opening held labels."""
import json
import sys
from pathlib import Path
import numpy as np
from data323 import ARMS, DATA_SHA, ORDERS, ROOT, sha


def verify(held,seed):
 p=ROOT/'output'/f'{held}_s{seed}';f=p/'result.npz'
 with np.load(f,allow_pickle=False) as z:
  m=json.loads(str(z['metadata']));idx=z['index']
  if set(z.files)!={'index','metadata'}|{f'{a}|{o}' for a in ARMS for o in ORDERS}:raise ValueError('fold matrix keys')
  if not len(idx) or np.any(np.diff(idx)<=0):raise ValueError('invalid held indices')
  for a in ARMS:
   for o in ORDERS:
    v=z[f'{a}|{o}']
    if v.shape!=(len(idx),3) or not np.isfinite(v).all() or not np.allclose(v.sum(1),1,atol=1e-6):raise ValueError('bad probabilities')
 if (m['held'],m['seed'])!=(held,seed) or m['held_labels_opened'] or len(m['fit_people'])!=9:
  raise ValueError('invalid source exclusion')
 if any(x.split('/')[-1]==held for x in m['fit_people']):raise ValueError('held person in fit')
 if m['source_manifest_sha256']!=sha(ROOT/'source323.sha256') or m['data_sha256']!=DATA_SHA or m['permutation_sha256']!=sha(ROOT/'runs/shuffle_perms.npz'):
  raise ValueError('source/data/permutation mismatch')
 if set(m['checkpoints'])!={'P0','P1','S0','S1'}:raise ValueError('checkpoint set differs')
 for name,digest in m['checkpoints'].items():
  if sha(p/f'{name}.pt')!=digest:raise ValueError('checkpoint hash differs')
 if len(set(m['initial_sha256'].values()))!=1:raise ValueError('initial weights differ')
 return {'held':held,'seed':seed,'output_sha256':sha(f),'seconds':m['elapsed_seconds'],'peak_gpu_bytes':m['peak_gpu_bytes']}

if __name__=='__main__':print(json.dumps(verify(sys.argv[1],int(sys.argv[2]))),flush=True)
