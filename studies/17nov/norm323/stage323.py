"""Stage all §323 producer seeds as exactly one averaged member per arm/order."""
import json
import os
from pathlib import Path
import numpy as np
from config import GROUPS, ORDERS, sha

PRODUCER=Path('/mnt/lustre/work/gehler/gfh527/ni26_s323_normsplit_20261117')
SOURCE='88437e69b5eaf7ff84a3635df88fe68b38394abbc527629b0f41c7fcc24eb6d5'
ARMS=('P0F','P0C','P1F','P1C','S0F','S0C','S1F','S1C')


def main():
 root=Path(os.environ['NI26_WS']).resolve(strict=True)
 manifest=PRODUCER/'source323.sha256';whole=PRODUCER/'whole_results.json';folder=PRODUCER/'runs/n323_streams'
 if sha(manifest)!=SOURCE or not whole.is_file():raise ValueError('complete locked whole producer required')
 result=json.loads(whole.read_text());validation=json.loads((folder/'validation.json').read_text())
 if result['source_manifest_sha256']!=SOURCE or validation['source_manifest']!=SOURCE or validation['folds']!=result['folds']:
  raise ValueError('producer source/fold identity differs')
 expected_folds={f'P{i:02d}_s{s}' for i in range(1,11) for s in (0,1)}
 if set(validation['folds'])!=expected_folds:raise ValueError('incomplete fold matrix')
 inputs={str(manifest):SOURCE,str(whole):sha(whole),str(folder/'validation.json'):sha(folder/'validation.json'),
         str(folder/'streams.sha256'):sha(folder/'streams.sha256')}
 for name,digest in validation['folds'].items():
  p=PRODUCER/'output'/name/'result.npz'
  if sha(p)!=digest:raise ValueError('fold changed')
  inputs[str(p)]=digest
  with np.load(p,allow_pickle=False) as z:m=json.loads(str(z['metadata']))
  for ck,h in m['checkpoints'].items():
   q=p.parent/f'{ck}.pt'
   if sha(q)!=h:raise ValueError('checkpoint changed')
   inputs[str(q)]=h
 with np.load(root/'data/proteus_windows.npz',allow_pickle=False) as z:n=len(z['subject'])
 expected={f'ttt18{o}_n323{a.lower()}_base-all_loso-s{s}.npz' for a in ARMS for s in (0,1) for o in ORDERS}
 if set(validation['streams'])!=expected:raise ValueError('incomplete 128 source streams')
 directory=root/'runs/representation_streams';target=root/'runs/representation_inputs'
 directory.mkdir(exist_ok=False);target.mkdir(exist_ok=False);records={}
 for a in ARMS:
  family='n323'+a.lower()
  for o in ORDERS:
   values=[]
   for seed in (0,1):
    name=f'ttt18{o}_{family}_base-all_loso-s{seed}.npz';p=folder/name;h=validation['streams'][name]
    if sha(p)!=h:raise ValueError('source stream changed')
    inputs[str(p)]=h
    with np.load(p,allow_pickle=False) as z:
     m=json.loads(str(z['metadata']))
     if (m['arm'],m['family'],m['seed'],m['order'],m['source_manifest'])!=(a,family,seed,o,SOURCE):raise ValueError('stream identity differs')
     if m['folds']!=validation['folds'] or not np.array_equal(z['index'],np.arange(n)):raise ValueError('stream fold/index differs')
     v=z['proba'].copy()
     if v.shape!=(n,3) or not np.isfinite(v).all() or (v<0).any() or not np.allclose(v.sum(1),1,atol=1e-6):raise ValueError('invalid probability')
     values.append(v)
   mean=np.mean(values,axis=0);meta=dict(family=family,order=o,seeds=[0,1],operation='arithmetic_probability_mean',producer_source=SOURCE)
   p=directory/f'ttt18{o}_{family}_base-all_loso-ensemble2.npz'
   with p.open('xb') as f:np.savez(f,index=np.arange(n),proba=mean,metadata=json.dumps(meta))
   records[p.name]=sha(p)
 output=target/'streams.sha256';output.write_text(''.join(f'{h}  {p}\n' for p,h in sorted(records.items())))
 (target/'lock.json').write_text(json.dumps(dict(manifest_sha256=sha(output),producer_inputs_sha256=inputs,group='323'),indent=2))
 print('complete §323 source matrix staged as 64 matched two-seed ensemble streams',flush=True)

if __name__=='__main__':main()
