"""Original complete-chain guards, then preregistered factorial interactions."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path('/mnt/lustre/work/gehler/gfh527/ni26_s323_chain_20261117')
OUT=Path('/mnt/lustre/work/gehler/gfh527/ni26_s323_interactions_20261117')
HERE=ROOT/'studies/14nov/newmember';sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('score323chain',HERE/'score_s_v66p.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
if Path(s.__file__).resolve()!=(HERE/'score_s_v66p.py').resolve():raise ValueError('wrong scorer module')
with np.load(ROOT/'runs/replay_data/bases.npz',allow_pickle=True) as z:
 sessions=len(np.unique(z['sess|all_loso']))
# All complete source, package, route, stream, zero and CP checks occur before loading labels.
outputs=s.collect(ROOT,2655,sessions)
with np.load(ROOT/'runs/replay_data/bases.npz',allow_pickle=True) as z:
 y=z['y|all_loso'];cell=z['cell|all_loso'];people=z['person|all_loso']
counts=np.array([len(np.unique(cell[people==p])) for p in s.PEOPLE]);arms=('P0F','P0C','P1F','P1C','S0F','S0C','S1F','S1C')
metric={(a,o):np.mean([s.person_accuracy(outputs[a,o,b],y,cell,people,s.PEOPLE) for b in s.BASES],axis=0) for a in arms for o in s.ORDERS}
forms={}
for z in (0,1):forms[f'train_x_inference_skip{z}']=[(f'S{z}C',1),(f'S{z}F',-1),(f'P{z}C',-1),(f'P{z}F',1)]
for i in 'FC':forms[f'train_x_skip_{i}']=[(f'S1{i}',1),(f'S0{i}',-1),(f'P1{i}',-1),(f'P0{i}',1)]
for t in 'PS':forms[f'inference_x_skip_{t}']=[(f'{t}1C',1),(f'{t}1F',-1),(f'{t}0C',-1),(f'{t}0F',1)]
forms['three_way']=forms['train_x_inference_skip1']+[(a,-w) for a,w in forms['train_x_inference_skip0']]
out={}
for name,terms in forms.items():
 d=np.stack([sum(w*metric[a,o] for a,w in terms) for o in s.ORDERS])
 out[name]={'terms':terms,'people_order':s.summarize(d,s.ORDERS,s.PEOPLE),'official_equal_cell':s.cell_weighted_summary(d,s.ORDERS,counts)}
result={'scope':'registered eight-arm full-system interactions; new-person routing emulation only',
 'source_manifest_sha256':s.sha256(ROOT/'representation-source.sha256'),
 'chain_report_sha256':s.sha256(ROOT/'results.txt'),'interactions':out}
with (OUT/'chain-interactions.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True)
print('OK: complete guarded §323 full-chain interactions',flush=True)
