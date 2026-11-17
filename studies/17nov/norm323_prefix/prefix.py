"""Post-hoc causal-BN contributions by true delivered session position."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

OWNER=Path('/mnt/lustre/work/gehler/gfh527/ni26_s323_interactions_20261117')
OUT=Path('/mnt/lustre/work/gehler/gfh527/ni26_s323_prefix_20261117')
source=OWNER/'interactions.py';guard_hash='900d16d028a8ff175a6c2db00b8a0b6b67b85d9bcdded31a5993f2da596b70a2'
import hashlib
if hashlib.sha256((OWNER/'interactions.sha256').read_bytes()).hexdigest()!=guard_hash:raise ValueError('original guard source lock changed')
import subprocess
subprocess.run(['sha256sum','-c','interactions.sha256'],cwd=OWNER,stdout=subprocess.DEVNULL,check=True)
text=source.read_text();anchor='# Held labels open only after complete original folds, checkpoints and streams have passed.'
if text.count(anchor)!=1:raise ValueError('original complete matrix guard changed')
env={'__name__':'prefix_guard','__file__':str(source)}
exec(compile(text.split(anchor,1)[0],str(source),'exec'),env)
ROOT,ORDERS,PEOPLE,pred=env['ROOT'],env['ORDERS'],env['PEOPLE'],env['pred']
import protocol, score_s_v66p as s, metrics
if Path(protocol.__file__).resolve()!=(ROOT/'experiments/protocol.py').resolve():raise ValueError('wrong protocol import')
if Path(s.__file__).resolve()!=(ROOT/'studies/14nov/newmember/score_s_v66p.py').resolve():raise ValueError('wrong scorer import')
d=protocol.load();y=d['y'];people=np.array([x.split('/')[-1] for x in d['subject']]);cell=np.char.add(np.char.add(d['stream'],'|'),d['interface'])
n=len(y);counts=np.array([len(np.unique(cell[people==p])) for p in PEOPLE]);positions={};coverage={}
with np.load(ROOT/'runs/shuffle_perms.npz',allow_pickle=False) as z:
 for o in ORDERS:
  perm=z[f'{o[:-1]}|{o[-1]}|all_loso'];rank=np.empty(n,int);rank[perm]=np.arange(n)
  if not np.array_equal(np.sort(perm),np.arange(n)):raise ValueError('invalid pinned order')
  age=np.zeros(n,int)
  for stream in np.unique(d['stream']):
   rows=np.flatnonzero(d['stream']==stream);rows=rows[np.argsort(rank[rows],kind='stable')];age[rows]=np.arange(1,len(rows)+1)
  if np.any(age<1):raise ValueError('session position missing')
  positions[o]=age
  cov={}
  for name,mask in (('1-10',age<=10),('11-40',(age>=11)&(age<=40)),('41+',age>=41)):
   total=missing=0
   for c in np.unique(cell):
    rows=(cell==c)
    for k in np.unique(y[rows]):
     total+=1;missing+=int(not np.any(rows & (y==k) & mask))
   cov[name]={'windows':int(mask.sum()),'full_cell_classes':total,'missing_cell_classes':missing}
  coverage[o]=cov
counts_pair={}
for causal,fixed in (('P0C','P0F'),('P1C','P1F'),('S0C','S0F'),('S1C','S1F')):
 bybin={key:{'up':[],'down':[]} for key in ('1-10','11-40','41+')}
 for o in ORDERS:
  age=positions[o];masks={'1-10':age<=10,'11-40':(age>=11)&(age<=40),'41+':age>=41}
  accum={key:{'up':[],'down':[]} for key in masks}
  for seed in (0,1):
   a=pred[causal,seed,o];b=pred[fixed,seed,o];pa=a.argmax(1);pb=b.argmax(1)
   expected_delta=s.person_accuracy(a,y,cell,people,PEOPLE)-s.person_accuracy(b,y,cell,people,PEOPLE)
   total=np.zeros(len(PEOPLE))
   for key,mask in masks.items():
    up=100*metrics.values_by_person(((pa==y)&(pb!=y)&mask).astype(float),y,cell,people,PEOPLE)
    down=100*metrics.values_by_person(((pa!=y)&(pb==y)&mask).astype(float),y,cell,people,PEOPLE)
    accum[key]['up'].append(up);accum[key]['down'].append(down);total+=up-down
   if not np.allclose(total,expected_delta,rtol=0,atol=1e-10):raise ValueError('bin contributions do not sum to original score')
  for key in masks:
   for kind in ('up','down'):bybin[key][kind].append(np.mean(accum[key][kind],axis=0))
 out={}
 for key,value in bybin.items():
  up=np.asarray(value['up']);down=np.asarray(value['down']);delta=up-down
  out[key]={'contribution_people_order':s.summarize(delta,ORDERS,PEOPLE),
            'contribution_official':s.cell_weighted_summary(delta,ORDERS,counts),
            'corrections_official':s.cell_weighted_summary(up,ORDERS,counts),
            'destructions_official':s.cell_weighted_summary(down,ORDERS,counts)}
 counts_pair[f'{causal}_minus_{fixed}']=out
result={'scope':'post-hoc selected whole-model effects by delivered participant/session position; existing causal method',
        'bins':'contributions use full-cell/full-class weights; Graz and BrainHero share session position',
        'whole_report_sha256':s.sha256(ROOT/'whole_results.json'),'guard_source_sha256':guard_hash,
        'coverage':coverage,'contrasts':counts_pair}
with (OUT/'prefix-results.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True)
print('OK: all guarded full-cell contributions sum to original paired effects',flush=True)
