"""Outer-fold labels are never read by the model producer."""
import hashlib
from pathlib import Path
import numpy as np
from source_labels import read_training_labels

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data/proteus_windows.npz'
DATA_SHA='2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327'
PEOPLE=tuple(f'P{i:02d}' for i in range(1,11))
ORDERS=('within0','within1','within2','within3','full0','full1','runs0','runs1')
ARMS=('P0F','P0C','P1F','P1C','S0F','S0C','S1F','S1C')


def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()


def load(held):
 if held not in PEOPLE or sha(DATA)!=DATA_SHA:raise ValueError('source identity differs')
 with np.load(DATA,allow_pickle=False) as z:
  subject=z['subject'].astype(str);x=z['X'].copy();session=z['session'].astype(str)
 ids=np.array([s.split('/')[-1] for s in subject]); fit=np.flatnonzero(ids!=held);test=np.flatnonzero(ids==held)
 if set(ids[fit])!=set(PEOPLE)-{held} or not len(test):raise ValueError('invalid outer split')
 labels=np.full(len(x),-1,dtype=np.int64);labels[fit]=read_training_labels(DATA,fit,len(x))
 if (labels[test]!=-1).any() or set(labels[fit])!={0,1,2}:raise ValueError('held labels loaded or missing fit class')
 stream=np.char.add(np.char.add(subject,'|'),session)
 return {'X':x,'y':labels,'subject':subject,'session':session,'stream':stream},fit,test
