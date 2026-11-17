"""Pinned original training/eight-order inference/selected-label/metric machinery."""
import sys
from pathlib import Path
import numpy as np

API=Path('/mnt/lustre/work/gehler/gfh527/ni26_reference_bridge336_v1_20261117')
sys.path.insert(0,str(API))
import recipe336 as original
if original.sha(API/'recipe336.py')!='468d632f51e52ffd16f1dff1ae61a2a4b22483a22a7618f25a6d36952d6df3be':
    raise ValueError('frozen native reference recipe changed')
BASE=original.BASE
PEOPLE=('P02','P03','P09');ARMS=('O','A','M','L','G')
CELLS=[(person,arm) for person in PEOPLE for arm in ARMS]
ORDERS=('within0','within1','within2','within3','full0','full1','runs0','runs1')


def source(person):
    if person not in PEOPLE:raise ValueError('unregistered held person')
    data,_,_,people=original.load();fit=np.flatnonzero(people!=person);held=np.flatnonzero(people==person)
    if len(set(people[fit]))!=9 or set(people[held])!={person}:raise ValueError('source9 exclusion failed')
    y=original.labels(fit,len(people));data['y']=np.full(len(people),-1,np.int64);data['y'][fit]=y
    data.update(w=np.ones(len(people)),nolow=np.zeros(len(people),bool),tag='')
    if np.any(data['y'][held]!=-1):raise ValueError('held labels reached source')
    return data,fit,held,people,y


def balanced(model,rows,rng,count):
    group=model._grp.cpu().numpy();sessions=model._row_sess;ids=np.unique(group[rows]);out=[]
    for _ in range(count):
        chosen=rng.choice(ids,2,replace=False);parts=[]
        for person in chosen:
            available=np.unique(sessions[rows[group[rows]==person]])
            session=rng.choice(available);candidate=rows[(group[rows]==person)&(sessions[rows]==session)]
            if len(candidate)<32:raise ValueError('source session smaller than32')
            parts.append(rng.choice(candidate,32,replace=False))
        batch=np.concatenate(parts)
        if len(np.unique(group[batch]))!=2:raise ValueError('balanced batch lacks distinct people')
        out.append(batch)
    return out
