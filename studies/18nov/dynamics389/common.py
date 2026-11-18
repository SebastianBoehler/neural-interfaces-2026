"""Pinned data and interfaces for a fresh source-excluded whole EEG decoder."""
import hashlib
from pathlib import Path
import sys
import numpy as np

WORK=Path('/mnt/lustre/work/gehler/gfh527')
ROOT=WORK/'ni26_s318_coral_fbl_20261117'
DATA=ROOT/'data/proteus_windows.npz'
DATA_SHA='2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327'
PERMS=ROOT/'runs/shuffle_perms.npz'
PERM_SHA='6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f'
OLD=WORK/'ni26_wave379_v1_20261118'
PEOPLE=tuple(f'P{i:02d}' for i in range(1,11))
SEEDS=(389,1389)
ORDERS=('within0','within1','within2','within3','full0','full1','runs0','runs1')
ARMS=('D','C','P','A','donor')
SENSORS=('FC3','FCz','FC4','C3','Cz','C4','CP3','CPz','CP4','P3','Pz','P4')


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def digest(v):
    return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()


def metadata():
    assert sha(DATA)==DATA_SHA and sha(PERMS)==PERM_SHA
    with np.load(DATA) as z:
        meta={k:z[k] for k in ('subject','session','interface')}
    meta['person']=np.array([s.rsplit('/',1)[-1] for s in meta['subject']])
    meta['stream']=np.char.add(np.char.add(meta['subject'],'|'),meta['session'])
    return meta


def orders():
    with np.load(PERMS) as z:
        out={o:z[o[:-1]+'|'+o[-1]+'|all_loso'] for o in ORDERS}
    assert all(np.array_equal(np.sort(v),np.arange(2655)) for v in out.values())
    return out
