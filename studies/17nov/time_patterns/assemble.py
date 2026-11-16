"""Publish only a complete validated 32-stream §314 matrix."""
import json
import sys
import os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from audit import ROOT,validate
from produce import ORDERS,sha


def main():
 d,result,folds=validate();n=len(d['subject']);out=ROOT/'runs/s314/streams'
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);files={}
 for arm,family in (('TF','tf314'),('TFD','tfd314')):
  for seed in (0,1):
   for order in ORDERS:
    path=out/f'ttt18{order}_{family}_base-all_loso-s{seed}.npz'
    m={'arm':arm,'seed':seed,'order':order,'fold_sha256':folds,'source_sha256':sha(ROOT/'s314-source.sha256')}
    with path.open('xb') as f:np.savez(f,index=np.arange(n,dtype=np.int64),proba=result[arm,seed,order],metadata=json.dumps(m,sort_keys=True))
    files[path.name]=sha(path)
 with (out/'streams.sha256').open('x') as f:
  for name,h in sorted(files.items()):print(h,name,file=f)
 with (out/'validation.json').open('x') as f:json.dump({'folds':folds,'streams':files,'source_sha256':sha(ROOT/'s314-source.sha256'),'held_accuracy_computed':False},f,sort_keys=True,indent=2)
 print('32 validated stateless target streams assembled; no accuracy read',flush=True)


if __name__=='__main__':main()
