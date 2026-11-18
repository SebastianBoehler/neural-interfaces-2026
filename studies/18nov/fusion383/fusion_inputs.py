"""Restore real final scores, their session-age domains and pinned native EEG actors."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

WORK=Path('/mnt/lustre/work/gehler/gfh527')
OWNER=WORK/'ni26_soft_csp361_v1_20261117'
COHORT=('P01','P02','P03','P09')
FITS={
    382:('ni26_joint382_actor_v1_20261118','2c1e872be6575e154ee59120f36d4c7085c1a3c1619efbb70abb18a56c6e9e77'),
    1382:('ni26_joint382_h100_1382_20261118/actor','0fdd714798bbd0cf7133f7e9002daa51869505f54bb673cb8c3b6e3a1be3efbb'),
}
MEMBERS=('J_real','J_blank','J_donor','K','F_real','A6','population')
BRANCHES=('JS','control')
WEIGHTS=(0.05,0.10)
sys.path.insert(0,str(OWNER))
from assets import CHAIN, chain_guard, pin, verify_sources
from measure import official


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restore():
    pin(OWNER/'source.sha256','9e889431c92f4759c79e0724446d19fc9503ac5b726464f73f9ed22b548f9ccd')
    for line in (OWNER/'source.sha256').read_text().splitlines():
        h,p=line.split(maxsplit=1)
        pin(OWNER/p,h)
    verify_sources()
    with np.load(CHAIN/'data/proteus_windows.npz') as d:
        meta={k:d[k] for k in ('subject','session','interface')}
    meta['person']=np.array([s.rsplit('/',1)[-1] for s in meta['subject']])
    meta['stream']=np.char.add(np.char.add(meta['subject'],'|'),meta['session'])
    sc,scores,people,_=chain_guard(meta)
    ages={}
    with np.load(CHAIN/'runs/shuffle_perms.npz') as z:
        for o in sc.ORDERS:
            perm=z[f'{o[:-1]}|{o[-1]}|all_loso']
            assert np.array_equal(np.sort(perm),np.arange(len(people)))
            age=np.empty(len(people),int)
            for s in np.unique(meta['stream']):
                rows=perm[meta['stream'][perm]==s]
                age[rows]=np.arange(len(rows))
            ages[o]=age
    return sc,scores,meta,ages


def probability(values,age):
    assert np.isfinite(values).all()
    raw=age<90
    assert np.array_equal((values>=0).all(1),raw) and (values[~raw]<=0).all()
    result=values.copy().astype(np.float64)
    assert np.allclose(result[raw].sum(1),1,atol=1e-5)
    x=result[~raw]
    result[~raw]=np.exp(x-x.max(1,keepdims=True))
    result/=result.sum(1,keepdims=True)
    assert np.array_equal(result.argmax(1),values.argmax(1))
    return result


def member_bank(seed,person,rows,orders):
    name,fixed=FITS[seed]
    root=WORK/name
    pin(root/'before_labels.json',fixed)
    record=json.loads((root/'before_labels.json').read_text())['actors'][person]
    assert record['held']==person and record['seed']==seed and record['held_labels_opened'] is False
    assert set(record['train_people']).isdisjoint(COHORT) and len(record['train_people'])==6
    assert record['test_rows']==hashlib.sha256(rows.tobytes()).hexdigest()
    pin(root/(person+'.npz'),record['outputs_sha256'])
    with np.load(root/(person+'.npz')) as z:
        assert np.array_equal(rows,z['index'])
        bank={ (a,o):z[a+'|'+o] for a in (*MEMBERS,'frozen','JS') for o in orders }
    for v in bank.values():
        assert v.shape==(len(rows),3) and np.isfinite(v).all() and (v>=0).all() and np.allclose(v.sum(1),1)
    return bank
