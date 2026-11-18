"""Seal every held prediction before opening evaluation labels and compute paired LOSO contrasts."""
import json
from pathlib import Path
import sys

import numpy as np

from score379 import BASE, CONTROL_SHA, DATA_SHA, METRICS, ORDER_SHA, ORDERS, PEOPLE
from score379 import controls_and_age, sha, source_manifest
from train import digest


def main():
    root = Path.cwd()
    manifest = source_manifest(root/'source.sha256')
    data = BASE/'data/proteus_windows.npz'
    assert sha(data) == DATA_SHA
    with np.load(data) as z:
        meta = {k:z[k] for k in ('subject', 'session', 'interface')}
    meta['stream'] = np.char.add(np.char.add(meta['subject'], '|'),meta['session'])
    people = np.array([s.split('/')[-1] for s in meta['subject']])
    cell = np.char.add(np.char.add(meta['stream'], '|'),meta['interface'])
    banks, _ = controls_and_age(meta)
    proof = {}
    for arm in ('real','phase'):
        banks[arm] = np.full((len(ORDERS),len(people)), -1)
    for p in PEOPLE:
        rows = np.flatnonzero(people==p)
        m = json.loads(Path(p+'.json').read_text())
        assert m['source_sha']==manifest and m['output_sha']==sha(Path(p+'.npz'))
        assert m['train_rows']==digest(np.flatnonzero(people!=p)) and set(m['source_people'])==set(PEOPLE)-{p}
        with np.load(p+'.npz') as z:
            assert np.array_equal(rows,z['index'])
            for arm in ('real','phase'):
                assert z[arm].shape == (len(rows),) and np.isin(z[arm],(0,1,2)).all()
                banks[arm][:,rows] = z[arm]
        proof[p]=m
    Path('before435.json').write_text(json.dumps(dict(folds=proof,orders=ORDER_SHA,controls=CONTROL_SHA),indent=2))
    assert sha(METRICS/'source.sha256')=='9e889431c92f4759c79e0724446d19fc9503ac5b726464f73f9ed22b548f9ccd'
    source_manifest(METRICS/'source.sha256')
    sys.path.insert(0,str(METRICS))
    from measure import official
    metric=official()
    scorer_dir=BASE/'studies/14nov/newmember'
    assert sha(scorer_dir/'score_s_v66p.py')=='ae387cd5a7b116fc1259001d4fe449ba1065b0cb5943c117d4f65ec87a0b40d4'
    sys.path.insert(0,str(scorer_dir))
    from score_s_v66p import summarize, cell_weighted_summary
    with np.load(data) as z:
        y = z['y']
    counts = np.array([len(set(cell[people==p])) for p in PEOPLE])
    assert counts.sum()==28
    tables = {}
    for arm, bank in banks.items():
        assert np.isin(bank,(0,1,2)).all()
        table=[]
        for decisions in bank:
            table.append([100*metric(y[people==p],decisions[people==p],meta,people==p)['balanced_accuracy_cell_mean']
                          for p in PEOPLE])
        tables[arm]=np.array(table)
    scores={a:float(np.average((v[:4].mean(0)+v[4:6].mean(0)+v[6:].mean(0))/3,weights=counts))
            for a,v in tables.items()}
    for arm,pin in [('real',44.11988710946),('frozen',76.27933502146),('JS',76.55173151211)]:
        assert abs(scores[arm]-pin)<1e-7
    delta=tables['phase']-tables['real']
    recalls={a:[float(100*np.mean([(banks[a][0,(cell==c)&(y==k)]==k).mean()
                                    for c in np.unique(cell)])) for k in range(3)] for a in ('real','phase')}
    summary=dict(scores=scores,official=cell_weighted_summary(delta,ORDERS,counts),
                 people_order=summarize(delta,ORDERS,PEOPLE), class_recalls=recalls,
                 changed_genuine_rows=int((banks['phase'][0]!=banks['real'][0]).sum()),
                 before_sha=sha(Path('before435.json')))
    Path('result435.json').write_text(json.dumps(summary,indent=2))
    print(summary,flush=True)


if __name__=='__main__':
    main()
