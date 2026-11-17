"""Verify the complete 20-fold matrix, then score whole models and stage streams."""
import hashlib
import importlib.util
import json
import os
import tempfile
from pathlib import Path
import numpy as np
from data323 import ARMS, DATA_SHA, ORDERS, PEOPLE, ROOT, sha
from verify_fold import verify


def main():
 source=sha(ROOT/'source323.sha256');expected={(p,s) for p in PEOPLE for s in (0,1)}
 found={p.name for p in (ROOT/'output').iterdir() if p.is_dir() and not p.name.startswith('.')}
 if found!={f'{p}_s{s}' for p,s in expected}:raise ValueError('incomplete fold directory matrix')
 with np.load(ROOT/'data/proteus_windows.npz',allow_pickle=False) as z:subjects=z['subject'].astype(str)
 n=len(subjects);merged={(a,s,o):np.full((n,3),np.nan) for a in ARMS for s in (0,1) for o in ORDERS};folds={}
 for person,seed in sorted(expected):
  result=verify(person,seed);path=ROOT/'output'/f'{person}_s{seed}'/'result.npz';folds[f'{person}_s{seed}']=result['output_sha256']
  with np.load(path,allow_pickle=False) as z:
   idx=z['index'];m=json.loads(str(z['metadata']))
   fit=np.flatnonzero(~np.char.endswith(subjects,'/'+person));test=np.flatnonzero(np.char.endswith(subjects,'/'+person))
   if not np.array_equal(idx,test) or m['source_rows_sha256']!=hashlib.sha256(fit.tobytes()).hexdigest():raise ValueError('fold rows differ')
   if m['held_rows_sha256']!=hashlib.sha256(test.tobytes()).hexdigest():raise ValueError('held rows differ')
   for a in ARMS:
    for o in ORDERS:merged[a,seed,o][idx]=z[f'{a}|{o}']
 if any(not np.isfinite(a).all() for a in merged.values()):raise ValueError('incomplete canonical predictions')
 # Labels open only after all 20 complete producer outputs and checkpoints passed verification.
 import protocol, score_s_v66p
 from score_s_v66p import person_accuracy, summarize, cell_weighted_summary
 if Path(score_s_v66p.__file__).resolve()!=(ROOT/'studies/14nov/newmember/score_s_v66p.py').resolve():raise ValueError('wrong scorer import')
 if Path(protocol.__file__).resolve()!=(ROOT/'experiments/protocol.py').resolve():raise ValueError('wrong protocol import')
 d=protocol.load();y=d['y'];person=np.array([x.split('/')[-1] for x in d['subject']]);cell=np.char.add(np.char.add(d['stream'],'|'),d['interface'])
 counts=np.array([len(np.unique(cell[person==p])) for p in PEOPLE]);mask=np.ones(n,bool)
 per={};official={}
 for a in ARMS:
  for seed in (0,1):
   for o in ORDERS:
    pred=merged[a,seed,o];per[a,seed,o]=person_accuracy(pred,y,cell,person,PEOPLE)
    official[a,seed,o]=100*protocol.score(y,pred.argmax(1),d,mask)['balanced_accuracy_cell_mean']
 metric={(a,o):np.mean([per[a,k,o] for k in (0,1)],axis=0) for a in ARMS for o in ORDERS}
 pairs=[('S0C','P0F'),('S0F','P0F'),('S0C','S0F'),('P0C','P0F')]
 pairs += [(f'{t}{skip}C',f'{t}{skip}F') for t in 'PS' for skip in (0,1)]
 pairs += [(f'S{skip}{i}',f'P{skip}{i}') for skip in (0,1) for i in 'FC']
 pairs += [(f'{t}1{i}',f'{t}0{i}') for t in 'PS' for i in 'FC']
 contrasts={}
 for a,b in dict.fromkeys(pairs):
  delta=np.stack([metric[a,o]-metric[b,o] for o in ORDERS])
  contrasts[f'{a}_minus_{b}']={'people_order':summarize(delta,ORDERS,PEOPLE),'official_equal_cell':cell_weighted_summary(delta,ORDERS,counts)}
 scheme=lambda vals:float(np.mean([np.mean([vals[o] for o in ORDERS if o.startswith(k)]) for k in ('within','full','runs')]))
 absolute={a:scheme({o:np.mean([official[a,k,o] for k in (0,1)]) for o in ORDERS}) for a in ARMS}
 out={'scope':'whole new-person alternatives; selected design, not independent confirmation or current-chain result',
      'seed_aggregation':'score each seed independently, then average metrics', 'source_manifest_sha256':source,
      'folds':folds,'official_equal_cell_scheme_equal':absolute,'contrasts':contrasts,
      'per_seed_person':{f'{a}_s{k}':summarize(np.stack([per[a,k,o] for o in ORDERS]),ORDERS,PEOPLE) for a in ARMS for k in (0,1)}}
 dest=ROOT/'runs/n323_streams'
 if dest.exists() or (ROOT/'whole_results.json').exists():raise FileExistsError('refusing output overwrite')
 temp=Path(tempfile.mkdtemp(prefix='.n323-streams-',dir=ROOT/'runs'));records={}
 for a in ARMS:
  for seed in (0,1):
   for o in ORDERS:
    family='n323'+a.lower();p=temp/f'ttt18{o}_{family}_base-all_loso-s{seed}.npz'
    meta=dict(arm=a,family=family,seed=seed,order=o,source_manifest=source,folds=folds)
    with p.open('xb') as f:np.savez_compressed(f,index=np.arange(n),proba=merged[a,seed,o],metadata=json.dumps(meta,sort_keys=True))
    records[p.name]=sha(p)
 (temp/'streams.sha256').write_text(''.join(f'{v}  {k}\n' for k,v in sorted(records.items())))
 (temp/'validation.json').write_text(json.dumps(dict(source_manifest=source,folds=folds,streams=records),sort_keys=True,indent=2))
 (ROOT/'whole_results.tmp.json').write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
 os.rename(temp,dest);os.rename(ROOT/'whole_results.tmp.json',ROOT/'whole_results.json')
 print('complete 20-fold guarded whole matrix scored; 128 canonical streams staged',flush=True)

if __name__=='__main__':main()
