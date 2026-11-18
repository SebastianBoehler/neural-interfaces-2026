"""Commit the two-seed prediction banks before loading held labels."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from score379 import BASE, CONTROL_SHA, METRICS, ORDERS, PEOPLE, ORDER_SHA, DATA_SHA
from score379 import controls_and_age, sha, source_manifest
from train import digest

ARMS = [a+'_'+h for a in ('ce','consistent') for h in ('real','zero','donor')]
PAIRS = [('consistent_real','ce_real')]+[(a+'_real',a+'_'+h) for a in ('ce','consistent') for h in ('zero','donor')]


def main():
    source = source_manifest(Path('source.sha256'))
    data = BASE/'data/proteus_windows.npz'
    assert sha(data)==DATA_SHA
    with np.load(data) as z:
        meta = {k:z[k] for k in ('subject','session','interface')}
    meta['stream'] = np.char.add(np.char.add(meta['subject'],'|'),meta['session'])
    cells = np.char.add(np.char.add(meta['stream'],'|'),meta['interface'])
    people = np.array([s.split('/')[-1] for s in meta['subject']])
    controls,_ = controls_and_age(meta)
    counts = np.array([len(set(cells[people==p])) for p in PEOPLE])
    assert counts.sum()==28
    proof,raw = {},{}
    for seed in (438,439):
        banks = {a:np.full((8,len(people)),-1,dtype=np.int8) for a in ARMS}
        for p in PEOPLE:
            name = f'{p}_{seed}'
            r = json.loads(Path(name+'.json').read_text())
            assert r['source']==source and r['outputs']==sha(Path(name+'.npz')) and r['orders']==ORDER_SHA
            assert r['seed']==seed and r['held']==p and set(r['source_people'])==set(PEOPLE)-{p}
            assert r['train_rows']==digest(np.flatnonzero(people!=p))
            proof[name]=r
            with np.load(name+'.npz') as z:
                assert np.array_equal(z['index'],np.flatnonzero(people==p))
                for a in ARMS:
                    for oi,o in enumerate(ORDERS):
                        pred=z[a+'|'+o]
                        assert pred.shape==z['index'].shape and np.isin(pred,(0,1,2)).all()
                        banks[a][oi,z['index']]=pred
        assert all(np.isin(v,(0,1,2)).all() for v in banks.values())
        raw[seed]={**banks,**controls}
    Path('before438.json').write_text(json.dumps(dict(folds=proof,controls=CONTROL_SHA,order=ORDER_SHA),indent=2))
    assert sha(METRICS/'source.sha256')=='9e889431c92f4759c79e0724446d19fc9503ac5b726464f73f9ed22b548f9ccd'
    source_manifest(METRICS/'source.sha256')
    sys.path.insert(0,str(METRICS))
    from measure import official
    metric=official()
    path=BASE/'studies/14nov/newmember/score_s_v66p.py'
    assert sha(path)=='ae387cd5a7b116fc1259001d4fe449ba1065b0cb5943c117d4f65ec87a0b40d4'
    spec=importlib.util.spec_from_file_location('intervals',path)
    scorer=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    with np.load(data) as z:
        y=z['y']
    tables={}
    for seed,banks in raw.items():
        tables[seed]={}
        for a,bank in banks.items():
            tables[seed][a]=np.array([[100*metric(y[people==p],v[people==p],meta,people==p)['balanced_accuracy_cell_mean']
                                      for p in PEOPLE] for v in bank])
    combined={a:(tables[438][a]+tables[439][a])/2 for a in tables[438]}
    result={}
    for seed,t in [*tables.items(),('mean',combined)]:
        scores={a:float(np.average((v[:4].mean(0)+v[4:6].mean(0)+v[6:].mean(0))/3,weights=counts)) for a,v in t.items()}
        assert abs(scores['frozen']-76.27933502146)<1e-7 and abs(scores['JS']-76.55173151211)<1e-7
        comparisons={}
        for a,b in PAIRS:
            d=t[a]-t[b]
            comparisons[a+'-'+b]=dict(official=scorer.cell_weighted_summary(d,ORDERS,counts),
                                      people_order=scorer.summarize(d,ORDERS,PEOPLE))
        result[str(seed)]=dict(scores=scores,comparisons=comparisons)
    result['before_sha']=sha(Path('before438.json'))
    result['class_usage']={s:{a:np.bincount(v.ravel(),minlength=3).tolist() for a,v in bank.items()} for s,bank in raw.items()}
    Path('result438.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
