"""Source-only preregistered factorial; no P01 label access or outer score."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
import protocol, session_bn
from data322 import DATA_SHA, FIT, ROOT, VAL, load, sha
from encoder312 import fingerprint
from model322 import Factor, setup, train, SEED, param_hash

if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):
 raise ValueError('registered A100 route required')
smoke=json.loads((ROOT/'smoke.json').read_text())
if not smoke['passed'] or smoke['source_manifest_sha256']!=sha(ROOT/'source322.sha256'):
 raise ValueError('matching no-score smoke required')
torch.set_num_threads(6);d,global_rows,fit,val=load(True)
if any(x.split('/')[-1] not in FIT for x in d['subject'][fit]):raise ValueError('bad fit subject')
x,y,scale,mu,sd,weights,groups=setup(d,fit,'cuda')
batch_hash=hashlib.sha256(b''.join(r.tobytes() for ep in groups for r in ep)).hexdigest()
orders=('within0','within1','within2','within3','full0','full1','runs0','runs1')
with np.load(ROOT/'runs/shuffle_perms.npz',allow_pickle=False) as z:
 perms={o:z[f'{o[:-1]}|{o[-1]}|all_loso'].copy() for o in orders}
for o,p in perms.items():
 if not np.array_equal(np.sort(p),np.arange(2655)):raise ValueError(f'bad permutation {o}')


def evaluate(model,reference=False):
 values={}
 for o in orders:
  rank=np.empty(2655,dtype=int);rank[perms[o]]=np.arange(2655)
  prediction=np.full((len(d['y']),3),np.nan)
  for stream in dict.fromkeys(d['stream'][val]):
   rows=val[d['stream'][val]==stream];rows=rows[np.argsort(rank[global_rows[rows]],kind='stable')]
   if reference:
    p=model.predict_proba_stream(rows);prefix=model.predict_proba_stream(rows[:min(10,len(rows))])
   else:
    with torch.no_grad():
     a=torch.as_tensor(d['X'][rows],device='cuda').float()/scale
     p=torch.softmax(model(a).double(),1).cpu().numpy()
     prefix=torch.softmax(model(a[:min(10,len(rows))]).double(),1).cpu().numpy()
   if not np.allclose(p[:len(prefix)],prefix,rtol=2e-5,atol=2e-5):raise ValueError('causal prefix differs')
   if p.shape!=(len(rows),3) or not np.isfinite(p).all() or not np.allclose(p.sum(1),1):raise ValueError('invalid probabilities')
   prediction[rows]=p
  by_person={}
  for person in VAL:
   mask=np.array([v.split('/')[-1]==person for v in d['subject']]);idx=np.flatnonzero(mask)
   by_person[person]=100*protocol.score(d['y'][idx],prediction[idx].argmax(1),d,mask)['balanced_accuracy_cell_mean']
  values[o]=by_person
 return values


results={};start=time.monotonic();initial={}
for norm in ('population','session'):
 for skip in (False,True):
  name=f'{norm}_{int(skip)}';model=Factor(norm,skip,'cuda');initial[name]=param_hash(model)
  info=train(model,x,y,mu,sd,weights,groups);before=fingerprint(model)
  out=evaluate(model)
  if fingerprint(model)!=before:raise ValueError('validation changed model state')
  results[name]=dict(training=info,score_by_order_person=out,model_sha256=before)
  del model
if len(set(initial.values()))!=1:raise ValueError('initial models differ')
reference=session_bn.make('fbcnet_raw_sbn',d,SEED)
if reference.align is not False or reference.epochs!=40:raise ValueError('raw reference changed')
reference.fit(fit);reference.set_k(20.);before=fingerprint(reference.net)
out=evaluate(reference,True)
if fingerprint(reference.net)!=before:raise ValueError('reference state mutated')
results['raw_reference']=dict(score_by_order_person=out,model_sha256=before)
summary={}
for name,r in results.items():
 vals=r['score_by_order_person']
 summary[name]={p:float(np.mean([np.mean([vals[o][p] for o in orders if o.startswith(k)])
                                      for k in ('within','full','runs')])) for p in VAL}
result=dict(scope='exploratory two-source-person validation; P01 excluded, no outer score',
 source_manifest_sha256=sha(ROOT/'source322.sha256'),data_sha256=DATA_SHA,
 permutation_sha256=sha(ROOT/'runs/shuffle_perms.npz'),source_fit_people=FIT,source_validation_people=VAL,
 fit_rows_sha256=hashlib.sha256(global_rows[fit].tobytes()).hexdigest(),
 validation_rows_sha256=hashlib.sha256(global_rows[val].tobytes()).hexdigest(),
 initial_sha256=initial,batches_sha256=batch_hash,source_scale=scale,seed=SEED,epochs=40,
 device=torch.cuda.get_device_name(0),seconds=time.monotonic()-start,summary=summary,arms=results)
with (ROOT/'pilot.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
