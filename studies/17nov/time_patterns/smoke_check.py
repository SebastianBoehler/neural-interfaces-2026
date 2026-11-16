"""Independently validate the complete P01 seed-0 real-data smoke, no scores."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from audit import ROOT,protocol
from produce import ORDERS,sha,digest
from features import TF_DIM,DYNAMIC_DIM


def main():
 path=ROOT/'runs/s314/folds/P01_s0_smoke.npz'
 with np.load(ROOT/'data/proteus_windows.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in ('subject','split')}
 held,tr,te=list(protocol.folds(d,'all_loso'))[0];test=np.flatnonzero(te);train=np.flatnonzero(tr)
 with np.load(path,allow_pickle=False) as z:
  m=json.loads(str(z['metadata'].item()));people=sorted(set(d['subject'][tr]));validation=people[0]
  ir=np.flatnonzero(tr&(d['subject']!=validation));vr=np.flatnonzero(tr&(d['subject']==validation))
  if (held,m['held'],m['seed'],m['smoke'],m['held_accuracy_computed'],m['source_people'],m['validation_person']) != ('P01','P01',0,True,False,people,validation):raise ValueError('smoke source exclusion failed')
  if (m['train_rows_sha256'],m['inner_fit_rows_sha256'],m['validation_rows_sha256'],m['test_rows_sha256']) != (digest(train),digest(ir),digest(vr),digest(test)):
   raise ValueError('smoke row hashes differ')
  if (m['data_sha256'],m['permutations_sha256'],m['source_sha256']) != (sha(ROOT/'data/proteus_windows.npz'),sha(ROOT/'runs/shuffle_perms.npz'),sha(ROOT/'s314-source.sha256')):
   raise ValueError('smoke input hashes differ')
  if digest(z['model|rms'])!=m['final_rms_sha256']:raise ValueError('smoke scaler mismatch')
  with np.load(ROOT/'runs/shuffle_perms.npz',allow_pickle=False) as perms:
   for arm,dim in (('TF',TF_DIM),('TFD',TF_DIM+DYNAMIC_DIM)):
    if z[f'model|{arm}|coef'].shape!=(3,dim):raise ValueError('smoke head shape differs')
    if arm=='TFD' and np.max(np.abs(z[f'model|{arm}|coef'][:,TF_DIM:]))<=1e-10:raise ValueError('smoke temporal features unused')
    p=z[arm]
    if p.shape!=(len(test),3) or not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-10):raise ValueError('smoke invalid probabilities')
    for order in ORDERS:
     perm=perms[f'{order[:-1]}|{order[-1]}|all_loso'].astype(np.int64);rows=perm[te[perm]];loc=np.searchsorted(test,rows)
     if not np.array_equal(z[f'{order}|index'],rows) or not np.array_equal(z[f'{order}|{arm}'],p[loc]):raise ValueError('smoke order mismatch')
  print('P01 source-only smoke passed; elapsed seconds',m['seconds'],'chosen C',m['chosen_C'],'no held score',flush=True)


if __name__=='__main__':main()
