"""Nested source-held calibration traces and genuine outer native k20 deployment traces."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.special import softmax

WORK=Path('/mnt/lustre/work/gehler/gfh527')
BODIES=WORK/'ni26_policy395_body_v1_20261118'
BODY_SHA='21fe76dea1fd0df489b02eea87d1c12b951687e0f3ab013ffa9ead5d64b13350'
DYN=WORK/'ni26_dynamics389_v1_20261118'
sys.path.insert(0,str(DYN))
from common import DATA,OLD,ORDERS,PEOPLE,ROOT,digest,metadata,orders,sha
assert sha(DYN/'source.sha256')=='fc284f7948aa6f4e43fb9d4299f452b3cb0c5330d0602e80f050239901adea97'
sys.path.insert(0,str(OLD))
assert sha(OLD/'source_labels.py')=='35edab56fe754e6c77a81f54f2010baca7e0912af55af9564043b2a8bde8c4dc'
from source_labels import read_training_labels
ASSEMBLY=WORK/'ni26_calibration339_setup_v1_20261117'
ARCH=WORK/'ni26_s323_normsplit_20261117'
VAL_SHA='d0386068e179d9491891897ce87f4e312c6db161014cd45451cfdfaee22c447a'


def source_bank(held,seed,meta,perms):
    assert sha(BODIES/'source.sha256')==BODY_SHA
    result=[]
    owned={}
    for block in range(10):
        excluded=tuple(PEOPLE[(block+j)%10] for j in (0,1,3,5))
        if held not in excluded:
            continue
        tag=f'b{block}_s{seed}'
        path=BODIES/(tag+'.npz')
        record=json.loads((BODIES/(tag+'.json')).read_text())
        test=np.flatnonzero(np.isin(meta['person'],excluded))
        fit=np.flatnonzero(~np.isin(meta['person'],excluded))
        assert record['source']==BODY_SHA and record['fit']==digest(fit) and record['test']==digest(test)
        assert record['epochs']==40 and record['seed']==seed and record['block']==block
        assert record['outputs']==sha(path) and tuple(record['excluded'])==excluded
        assert record['held_labels_opened'] is False and set(record['fit_people'])==set(PEOPLE)-set(excluded)
        owned[tag]=record['outputs']
        with np.load(path) as z:
            assert np.array_equal(z['index'],test)
            for who in excluded:
                if who==held:
                    continue
                for o,p in perms.items():
                    lookup=softmax(z[o],axis=1)
                    for stream in np.unique(meta['stream'][meta['person']==who]):
                        rows=p[meta['stream'][p]==stream]
                        assert not np.intersect1d(rows,fit).size
                        result.append((who,o,stream,rows,lookup[np.searchsorted(test,rows)]))
    assert len(owned)==4 and set(v[0] for v in result)==set(PEOPLE)-{held}
    return result,owned


def outer(held,meta):
    assert sha(ASSEMBLY/'validation.json')==VAL_SHA
    proof=json.loads((ASSEMBLY/'validation.json').read_text())
    assert proof['raw_FB_bitwise323']
    target=np.flatnonzero(meta['person']==held)
    branches={}
    for seed in (0,1):
        path=ARCH/'output'/f'{held}_s{seed}'/'result.npz'
        assert sha(path)==proof['inputs'][str(path)]
        with np.load(path) as z:
            assert np.array_equal(z['index'],target)
            branches[seed]={k+'|'+o:z[stored+'|'+o] for k,stored in [('native','P0C'),('static','P0F')] for o in ORDERS}
    for name,old in [('native','rb'),('static','rf')]:
        for o in ORDERS:
            path=ASSEMBLY/'streams'/f'ttt18{o}_n339{old}_base-all_loso-ensemble2.npz'
            assert sha(path)==proof['streams'][path.name]
            with np.load(path) as z:
                assert np.array_equal(z['index'],np.arange(len(meta['person'])))
                ref=z['proba'][target]
            assert np.array_equal(np.mean([branches[s][name+'|'+o] for s in (0,1)],0),ref)
    return target,branches
