"""Validate the complete §314 source, model, fold and stateless stream matrix."""
import json
import os
import sys
from pathlib import Path
import numpy as np

ROOT=Path(os.environ['NI26_WS']).resolve(strict=True)
sys.path[:0]=[str(Path(__file__).resolve().parent),str(ROOT/'experiments')]
import protocol
from produce import ORDERS,sha,digest
from features import TF_DIM,DYNAMIC_DIM


def validate():
 with np.load(ROOT/'data/proteus_windows.npz',allow_pickle=False) as z:
  d={k:z[k].copy() for k in ('subject','session','interface','split')}
 n=len(d['subject']);d['stream']=np.char.add(np.char.add(d['subject'],'|'),d['session'])
 folds=list(protocol.folds(d,'all_loso'));people=[f'P{i:02d}' for i in range(1,11)]
 if [x[0] for x in folds]!=people:raise ValueError('unexpected fold order')
 expected={f'{p}_s{s}.npz' for p in people for s in (0,1)}
 found={p.name for p in (ROOT/'runs/s314/folds').glob('*.npz')}
 if found!=expected:raise ValueError('incomplete or extra fold matrix')
 with np.load(ROOT/'runs/shuffle_perms.npz',allow_pickle=False) as z:
  perms={o:z[f'{o[:-1]}|{o[-1]}|all_loso'].astype(np.int64) for o in ORDERS}
 results={(a,s,o):np.full((n,3),np.nan) for a in ('TF','TFD') for s in (0,1) for o in ORDERS}
 fold_hashes={};source_hash=sha(ROOT/'s314-source.sha256')
 for held,tr,te in folds:
  train,test=np.flatnonzero(tr),np.flatnonzero(te);fit_people=sorted(set(d['subject'][tr]))
  for seed in (0,1):
   path=ROOT/f'runs/s314/folds/{held}_s{seed}.npz';before=sha(path)
   with np.load(path,allow_pickle=False) as z:
    m=json.loads(str(z['metadata'].item()));validation=fit_people[seed]
    ir=np.flatnonzero(tr & (d['subject']!=validation));vr=np.flatnonzero(tr & (d['subject']==validation))
    if (m['held'],m['seed'],m['source_people'],m['validation_person'],m['smoke'],m['held_accuracy_computed']) != (held,seed,fit_people,validation,False,False):
     raise ValueError('fold identity or source-only guard failed')
    if (m['train_rows_sha256'],m['inner_fit_rows_sha256'],m['validation_rows_sha256'],m['test_rows_sha256']) != (digest(train),digest(ir),digest(vr),digest(test)):
     raise ValueError('fold row fingerprint changed')
    if (m['data_sha256'],m['permutations_sha256'],m['source_sha256']) != (sha(ROOT/'data/proteus_windows.npz'),sha(ROOT/'runs/shuffle_perms.npz'),source_hash):
     raise ValueError('source/data/permutation provenance differs')
    if digest(z['model|rms'])!=m['final_rms_sha256']:raise ValueError('fitted RMS changed')
    for arm in ('TF','TFD'):
     dim=TF_DIM if arm=='TF' else TF_DIM+DYNAMIC_DIM; model=m['model'][arm]
     if model['features']!=dim or m['chosen_C'][arm] not in (.01,.1,1.):raise ValueError('wrong supervised recipe')
     for key,shape in (('coef',(3,dim)),('intercept',(3,)),('scaler_mean',(dim,)),('scaler_scale',(dim,))):
      a=z[f'model|{arm}|{key}']
      if a.shape!=shape or not np.isfinite(a).all() or digest(a)!=model[f'{key}_sha256']:raise ValueError('model fingerprint failed')
     if arm=='TFD' and np.max(np.abs(z['model|TFD|coef'][:,TF_DIM:]))<=1e-10:raise ValueError('inactive temporal bank')
     p=z[arm]
     if p.shape!=(len(test),3) or not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(1),1,atol=1e-10):raise ValueError('invalid prediction')
     for o,perm in perms.items():
      if not np.array_equal(np.sort(perm),np.arange(n)):raise ValueError('invalid permutation')
      indices=perm[te[perm]];loc=np.searchsorted(test,indices)
      if not np.array_equal(z[f'{o}|index'],indices) or not np.array_equal(z[f'{o}|{arm}'],p[loc]):raise ValueError('stateless shuffled order differs')
      results[arm,seed,o][indices]=z[f'{o}|{arm}']
   if sha(path)!=before:raise ValueError('fold changed while validating')
   fold_hashes[path.name]=before
 if any(not np.isfinite(x).all() for x in results.values()):raise ValueError('unfilled held rows')
 return d,results,fold_hashes
