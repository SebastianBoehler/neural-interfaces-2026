"""Full-matrix per-seed/person audit and unlabeled source-only logit decomposition."""
import importlib.util
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/mnt/lustre/work/gehler/gfh527/ni26_raw312_20261117_matrix2')
OUT=Path('/mnt/lustre/work/gehler/gfh527/ni26_raw312_gap_audit_20261117')
if os.environ.get('NI26_WS')!=str(ROOT):raise ValueError('wrong producer root')
sys.path[:0]=[str(ROOT),str(ROOT/'experiments'),str(ROOT/'studies/14nov/newmember')]
import protocol
from data import sha
from encoder import Encoder, fingerprint, moments
path=ROOT/'studies/14nov/newmember/score_s_v66p.py'
spec=importlib.util.spec_from_file_location('score312audit',path)
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
ORDERS=('within0','within1','within2','within3','full0','full1','runs0','runs1')
PEOPLE=tuple(f'P{i:02d}' for i in range(1,11));ARMS=('S','F','T','C')
report=ROOT/'whole_results.json'
if sha(report)!='1030e86966f7b7a1d9cade0fb60dafd2c00e29d159430b774c504878cec8db5b':raise ValueError('whole report changed')
result=json.loads(report.read_text());hashes=result['hashes'];metadata={}
data_hash=sha(ROOT/'data/proteus_windows.npz'); perm_hash=sha(ROOT/'runs/shuffle_perms.npz')
if Path(protocol.__file__).resolve() != (ROOT/'experiments/protocol.py').resolve():raise ValueError('protocol module mismatch')
for a in ('fb','sh'):
 for arm in ARMS:
  for seed in (0,1):
   for person in PEOPLE:
    name=f'{a}_{arm}_s{seed}_{person}.npz';p=ROOT/'output'/name
    if sha(p)!=hashes[name]:raise ValueError('matrix hash changed')
    with np.load(p,allow_pickle=False) as z:m=json.loads(str(z['metadata']))
    if sha(p.with_suffix('.pt'))!=m['checkpoint_sha256']:raise ValueError('checkpoint changed')
    if (m['held'],m['architecture'],m['arm'],m['seed'])!=(person,a,arm,seed):raise ValueError('identity differs')
    if m['source_manifest']!=sha(ROOT/'source.sha256') or len(m['fit_people'])!=9 or len(m['inner_fit_people'])!=8:raise ValueError('source guard')
    if m['data_sha256']!=data_hash or m['permutation_sha256']!=perm_hash:raise ValueError('data/permutation differs')
    if any(q.split('/')[-1]==person for q in m['fit_people']):raise ValueError('held in fit')
    metadata[name]=m
# The owning complete collector has already verified all 160 entries before opening labels.
d=protocol.load(); y=d['y']; subjects=d['subject']; person=np.array([x.split('/')[-1] for x in subjects])
cell=np.char.add(np.char.add(d['stream'],'|'),d['interface']); n=len(y)
import hashlib
for name,m in metadata.items():
 rows=np.flatnonzero(np.isin(subjects,m['fit_people']))
 if hashlib.sha256(rows.tobytes()).hexdigest()!=m['source_rows_sha256']:raise ValueError(f'fit rows differ: {name}')
all_results={}; epochs={}
for a in ('fb','sh'):
 for arm in ARMS:
  for seed in (0,1):
   ordered={o:np.full((n,3),np.nan) for o in ORDERS}
   for held in PEOPLE:
    name=f'{a}_{arm}_s{seed}_{held}.npz';p=ROOT/'output'/name;m=metadata[name]
    with np.load(p,allow_pickle=False) as z:
     rows=np.flatnonzero(person==held)
     if not np.array_equal(z['index'],rows):raise ValueError('canonical held rows differ')
     for o in ORDERS:ordered[o][rows]=z[f'proba_{o}']
    epochs[name]={'selected':m['selected_epochs'],'validation':m['validation_person']}
   metrics=[]
   for o in ORDERS:
    if not np.isfinite(ordered[o]).all():raise ValueError('nonfinite complete matrix')
    metrics.append(s.person_accuracy(ordered[o],y,cell,person,PEOPLE))
   x=np.asarray(metrics)
   scheme=np.mean([x[:4].mean(0),x[4:6].mean(0),x[6:].mean(0)],0)
   all_results[f'{a}_{arm}_s{seed}']=dict(zip(PEOPLE,map(float,scheme)))
# Cross-check exact existing complete-collector T-C people effects.
for a in ('fb','sh'):
 for p in PEOPLE:
  measured=np.mean([all_results[f'{a}_T_s{k}'][p]-all_results[f'{a}_C_s{k}'][p] for k in (0,1)])
  expected=result['representations'][a]['effects']['T_minus_C']['people'][p]
  if not np.isclose(measured,expected,rtol=0,atol=1e-9):raise ValueError('per-seed metric does not match collector')
# Source rows only; output contributions are descriptive, not independent validation.
torch.set_num_threads(2); torch.use_deterministic_algorithms(False)
contributions={}
for a in ('fb','sh'):
 for arm in ('S','T'):
  for seed in (0,1):
   name=f'{a}_{arm}_s{seed}_P01.npz';m=metadata[name]
   rows=np.concatenate([np.flatnonzero(subjects==p)[:8] for p in m['fit_people']])
   if len(rows)!=72 or any(person[i]=='P01' for i in rows):raise ValueError('source-only rows guard')
   ck=torch.load((ROOT/'output'/name).with_suffix('.pt'),map_location='cpu',weights_only=True)
   net=Encoder(a,seed,'cpu');net.load_state_dict(ck['state']);net.eval()
   before=fingerprint(net);x=torch.as_tensor(d['X'][rows]).float()/float(ck['scale'])
   with torch.no_grad():
    z=net.representation(x); skip=(moments(x)-net.moment_mean)/net.moment_sd
    w=net.task.weight;body=z@w[:,:z.shape[1]].T; residual=skip@w[:,z.shape[1]:].T
    center=lambda t:t-t.mean(1,keepdim=True)
    b=float(center(body).square().mean().sqrt());q=float(center(residual).square().mean().sqrt())
    full=torch.softmax(body+residual+net.task.bias,1)
    no_body=torch.softmax(residual+net.task.bias,1);no_skip=torch.softmax(body+net.task.bias,1)
    contributions[name]={'body_logit_rms':b,'skip_logit_rms':q,'body_to_skip_rms':b/q,
                         'mean_abs_prob_change_without_body':float((full-no_body).abs().mean()),
                         'mean_abs_prob_change_without_skip':float((full-no_skip).abs().mean()),
                         'source_rows':len(rows)}
   if before!=fingerprint(net):raise ValueError('model state changed')
result_out={'scope':'post-hoc whole-decoder per-seed/person; unlabeled source-only checkpoint contributions; no target ablation',
            'source_report_sha256':sha(report),'per_seed_people':all_results,'epochs':epochs,'source_contributions':contributions}
dest=OUT/'gap-audit.json'
with dest.open('x') as f:json.dump(result_out,f,indent=2)
print('complete matrix per-person/seed and source-only contribution audit OK',flush=True)
