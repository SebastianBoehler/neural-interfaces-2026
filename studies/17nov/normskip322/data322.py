"""Source-person-only dataset; outer P01 labels are never opened."""
import hashlib
from pathlib import Path
import numpy as np
from source_labels import read_selected

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data/proteus_windows.npz'
DATA_SHA='2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327'
FIT=tuple(f'P{i:02}' for i in range(4,11));VAL=('P02','P03')


def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()


def load(validation_labels=False):
 if sha(DATA)!=DATA_SHA:raise ValueError('official tensor hash differs')
 with np.load(DATA,allow_pickle=False) as z:
  subject=z['subject'].astype(str); total=len(subject)
  ids=np.array([x.split('/')[-1] for x in subject]);rows=np.flatnonzero(np.isin(ids,FIT+VAL))
  d={k:z[k][rows] for k in ('X','subject','session','run','split')}
  d['ch_names']=z['ch_names'].copy()
 if set(ids[rows])!=set(FIT+VAL) or np.any(ids[rows]=='P01'):raise ValueError('source person scope differs')
 d['stream']=np.char.add(np.char.add(d['subject'],'|'),d['session'])
 d['w']=np.ones(len(rows));d['nolow']=np.zeros(len(rows),bool);d['tag']=''
 local_ids=ids[rows];fit=np.flatnonzero(np.isin(local_ids,FIT));val=np.flatnonzero(np.isin(local_ids,VAL))
 selected=np.sort(np.r_[fit,val] if validation_labels else fit)
 d['y']=np.full(len(rows),-1,dtype=np.int64);d['interface']=np.full(len(rows),'',dtype='<U32')
 d['y'][selected]=read_selected(DATA,'y',rows[selected],total)
 d['interface'][selected]=read_selected(DATA,'interface',rows[selected],total)
 if set(d['y'][fit])!={0,1,2}:raise ValueError('fit class missing')
 if not validation_labels and np.any(d['y'][val]!=-1):raise ValueError('validation labels opened during smoke')
 return d,rows,fit,val
