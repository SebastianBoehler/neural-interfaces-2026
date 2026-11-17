"""Fit four source-only models and stream their eight matched inference arms."""
import copy
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
import model322, data323, modes, encoder312, session_bn, source_labels
from encoder312 import fingerprint
from data323 import ARMS, DATA_SHA, ORDERS, PEOPLE, ROOT, load, sha
from model322 import Factor, setup, train, param_hash
from modes import probabilities, views


def run(held,seed):
 if held not in PEOPLE or seed not in (0,1):raise ValueError('invalid registered fold')
 if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):raise ValueError('A100 required')
 for mod,expected in ((model322,ROOT/'model322.py'),(data323,ROOT/'data323.py'),(modes,ROOT/'modes.py'),
                      (encoder312,ROOT/'encoder312.py'),(session_bn,ROOT/'experiments/session_bn.py'),
                      (source_labels,ROOT/'source_labels.py')):
  if Path(mod.__file__).resolve()!=expected.resolve():raise ValueError(f'wrong module path: {mod.__name__}')
 torch.set_num_threads(6);torch.cuda.reset_peak_memory_stats();start=time.monotonic()
 dest=ROOT/'output'/f'{held}_s{seed}';dest.parent.mkdir(exist_ok=True)
 if dest.exists():raise FileExistsError(dest)
 tmp=Path(tempfile.mkdtemp(prefix=f'.{held}_s{seed}_',dir=dest.parent))
 d,fit,test=load(held);model322.SEED=seed
 x,y,scale,mu,sd,weights,groups=setup(d,fit,'cuda')
 batch_hash=hashlib.sha256(b''.join(b.tobytes() for ep in groups for b in ep)).hexdigest()
 with np.load(ROOT/'runs/shuffle_perms.npz',allow_pickle=False) as z:
  perms={o:z[f'{o[:-1]}|{o[-1]}|all_loso'].copy() for o in ORDERS}
 for o,p in perms.items():
  if not np.array_equal(np.sort(p),np.arange(len(d['X']))):raise ValueError(f'bad permutation {o}')
 result={};initial={};checkpoints={};details={}
 for norm,label in (('population','P'),('session','S')):
  for skip in (False,True):
   base=f'{label}{int(skip)}';model=Factor(norm,skip,'cuda');initial[base]=param_hash(model)
   probe=copy.deepcopy(model).train();probe.moment_mean.copy_(mu);probe.moment_sd.copy_(sd)
   probe.skip=skip;torch.manual_seed(seed);probe.zero_grad(set_to_none=True)
   probe_rows=groups[0][0];loss=torch.nn.functional.cross_entropy(probe(x[probe_rows]),y[probe_rows],weight=weights)
   loss.backward();g=probe.task.weight.grad[:,-287:]
   if not torch.isfinite(g).all() or (skip and float(g.abs().max())<=0) or (not skip and float(g.abs().max())!=0):
    raise ValueError('skip active/zero gradient guard failed')
   del probe
   details[base]=train(model,x,y,mu,sd,weights,groups)
   model.eval();original=fingerprint(model)
   ck=tmp/f'{base}.pt';torch.save(dict(state=model.state_dict(),scale=scale,norm=norm,skip=skip,seed=seed,held=held),ck)
   checkpoints[base]=sha(ck)
   fixed,causal=views(model,x[:min(16,len(x))]);view_models={'F':fixed,'C':causal}
   view_hash={k:fingerprint(v) for k,v in view_models.items()}
   for mode,view in view_models.items():
    arm=base+mode
    for order,p in perms.items():
     rank=np.empty(len(p),int);rank[p]=np.arange(len(p));arr=np.full((len(test),3),np.nan)
     pos=np.full(len(d['X']),-1,int);pos[test]=np.arange(len(test))
     for session in dict.fromkeys(d['stream'][test]):
      rows=test[d['stream'][test]==session];rows=rows[np.argsort(rank[rows],kind='stable')]
      q=probabilities(view,d['X'][rows]/scale,'cuda')
      prefix=probabilities(view,d['X'][rows[:min(10,len(rows))]]/scale,'cuda')
      if not np.allclose(q[:len(prefix)],prefix,rtol=2e-5,atol=2e-5):raise ValueError('prefix differs')
      other=test[d['stream'][test]!=session]
      if len(other):
       probabilities(view,d['X'][other[:min(10,len(other))]]/scale,'cuda')
       repeated=probabilities(view,d['X'][rows[:min(10,len(rows))]]/scale,'cuda')
       if not np.array_equal(prefix,repeated):raise ValueError('session reset/interleaving differs')
      arr[pos[rows]]=q
     if not np.isfinite(arr).all() or not np.allclose(arr.sum(1),1,atol=1e-6):raise ValueError('bad predictions')
     result[f'{arm}|{order}']=arr
    if fingerprint(view)!=view_hash[mode]:raise ValueError('inference changed model state')
   if fingerprint(model)!=original:raise ValueError('fitted model state changed')
   for order in ORDERS[1:]:
    first=result[f'{base}F|{ORDERS[0]}'];other=result[f'{base}F|{order}']
    top=np.sort(first,axis=1);valid=top[:,-1]-top[:,-2]>2e-5
    if not np.array_equal(first.argmax(1)[valid],other.argmax(1)[valid]):raise ValueError('fixed inference order changed labels')
   if not any(np.max(np.abs(result[f'{base}F|{o}']-result[f'{base}C|{o}']))>1e-7 for o in ORDERS):
    raise ValueError('causal normalization inactive')
   del model,fixed,causal
 if len(set(initial.values()))!=1:raise ValueError('same-seed model initializations differ')
 if set(result)!={f'{a}|{o}' for a in ARMS for o in ORDERS}:raise ValueError('incomplete output matrix')
 meta=dict(held=held,seed=seed,fit_people=sorted(set(d['subject'][fit].tolist())),
  source_rows_sha256=hashlib.sha256(fit.tobytes()).hexdigest(),held_rows_sha256=hashlib.sha256(test.tobytes()).hexdigest(),
  source_manifest_sha256=sha(ROOT/'source323.sha256'),data_sha256=DATA_SHA,
  permutation_sha256=sha(ROOT/'runs/shuffle_perms.npz'),batches_sha256=batch_hash,
  initial_sha256=initial,checkpoints=checkpoints,training=details,scale=scale,
  device=torch.cuda.get_device_name(0),peak_gpu_bytes=torch.cuda.max_memory_allocated(),
  elapsed_seconds=time.monotonic()-start,held_labels_opened=False,session_policy='causal and independently reset')
 if any(p.split('/')[-1]==held for p in meta['fit_people']) or len(meta['fit_people'])!=9:raise ValueError('held source leakage')
 with (tmp/'result.npz').open('xb') as f:np.savez_compressed(f,index=test,metadata=json.dumps(meta),**result)
 os.rename(tmp,dest)
 print(json.dumps({'held':held,'seed':seed,'fold':str(dest),'elapsed_seconds':meta['elapsed_seconds'],
                   'peak_gpu_bytes':meta['peak_gpu_bytes'],'source_sha256':meta['source_manifest_sha256']}),flush=True)

if __name__=='__main__':run(sys.argv[1],int(sys.argv[2]))
