"""Registered complete-matrix factorial interactions, using original metrics."""
import json
import os
import sys
from pathlib import Path
import numpy as np
ROOT=Path('/mnt/lustre/work/gehler/gfh527/ni26_s323_normsplit_20261117')
OUT=Path('/mnt/lustre/work/gehler/gfh527/ni26_s323_interactions_20261117')
sys.path[:0]=[str(ROOT),str(ROOT/'experiments'),str(ROOT/'studies/14nov/newmember')]
from data323 import ARMS, ORDERS, PEOPLE, sha
from verify_fold import verify
SOURCE='88437e69b5eaf7ff84a3635df88fe68b38394abbc527629b0f41c7fcc24eb6d5'
if sha(ROOT/'source323.sha256')!=SOURCE:raise ValueError('source changed')
whole=json.loads((ROOT/'whole_results.json').read_text());folder=ROOT/'runs/n323_streams'
v=json.loads((folder/'validation.json').read_text())
if whole['source_manifest_sha256']!=SOURCE or v['source_manifest']!=SOURCE or whole['folds']!=v['folds']:
 raise ValueError('complete report identity differs')
if set(v['folds'])!={f'{p}_s{s}' for p in PEOPLE for s in (0,1)}:raise ValueError('fold matrix incomplete')
for p in PEOPLE:
 for seed in (0,1):
  result=verify(p,seed)
  if result['output_sha256']!=v['folds'][f'{p}_s{seed}']:raise ValueError('fold changed')
expected={f'ttt18{o}_n323{a.lower()}_base-all_loso-s{k}.npz' for a in ARMS for k in (0,1) for o in ORDERS}
if set(v['streams'])!=expected:raise ValueError('stream matrix incomplete')
with np.load(ROOT/'data/proteus_windows.npz',allow_pickle=False) as z:subjects=z['subject'].astype(str)
n=len(subjects);pred={}
for a in ARMS:
 for k in (0,1):
  for o in ORDERS:
   name=f'ttt18{o}_n323{a.lower()}_base-all_loso-s{k}.npz';path=folder/name
   if sha(path)!=v['streams'][name]:raise ValueError('stream changed')
   with np.load(path,allow_pickle=False) as z:
    info=json.loads(str(z['metadata']));p=z['proba'].copy()
    if (info['arm'],info['seed'],info['order'],info['source_manifest'])!=(a,k,o,SOURCE):raise ValueError('stream identity')
    if info['folds']!=v['folds'] or not np.array_equal(z['index'],np.arange(n)):raise ValueError('stream rows/folds')
    if p.shape!=(n,3) or not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-6):raise ValueError('bad probability')
    pred[a,k,o]=p
# Held labels open only after complete original folds, checkpoints and streams have passed.
import protocol, score_s_v66p as s
if Path(protocol.__file__).resolve()!=(ROOT/'experiments/protocol.py').resolve():raise ValueError('wrong protocol')
if Path(s.__file__).resolve()!=(ROOT/'studies/14nov/newmember/score_s_v66p.py').resolve():raise ValueError('wrong scorer')
d=protocol.load();y=d['y'];people=np.array([x.split('/')[-1] for x in d['subject']]);cell=np.char.add(np.char.add(d['stream'],'|'),d['interface'])
counts=np.array([len(np.unique(cell[people==p])) for p in PEOPLE])
metric={(a,o):np.mean([s.person_accuracy(pred[a,k,o],y,cell,people,PEOPLE) for k in (0,1)],axis=0) for a in ARMS for o in ORDERS}
forms={}
for z in (0,1):forms[f'train_x_inference_skip{z}']=[(f'S{z}C',1),(f'S{z}F',-1),(f'P{z}C',-1),(f'P{z}F',1)]
for i in 'FC':forms[f'train_x_skip_{i}']=[(f'S1{i}',1),(f'S0{i}',-1),(f'P1{i}',-1),(f'P0{i}',1)]
for t in 'PS':forms[f'inference_x_skip_{t}']=[(f'{t}1C',1),(f'{t}1F',-1),(f'{t}0C',-1),(f'{t}0F',1)]
forms['three_way'] = forms['train_x_inference_skip1']+[(a,-w) for a,w in forms['train_x_inference_skip0']]
out={}
for name,terms in forms.items():
 delta=np.stack([sum(w*metric[a,o] for a,w in terms) for o in ORDERS])
 out[name]={'terms':terms,'people_order':s.summarize(delta,ORDERS,PEOPLE),'official_equal_cell':s.cell_weighted_summary(delta,ORDERS,counts)}
result={'scope':'pre-registered complete whole-decoder factorial interactions; selected design, not independent validation',
        'source_manifest_sha256':SOURCE,'whole_report_sha256':sha(ROOT/'whole_results.json'),'interactions':out}
with (OUT/'interactions.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True)
print('OK: complete guarded registered factorial interactions',flush=True)
