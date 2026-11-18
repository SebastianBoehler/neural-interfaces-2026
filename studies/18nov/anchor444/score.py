"""Evaluate paired full decoder outputs using the registered people/order estimator."""
import json
import os
from pathlib import Path
import sys

import numpy as np
import score379 as ref


def main():
    meta, people, rows = {},ref.PEOPLE,ref.ORDERS
    data = ref.BASE/'data/proteus_windows.npz'
    assert ref.sha(data)==ref.DATA_SHA
    with np.load(data) as z:
        meta={k:z[k] for k in ('subject','session','interface')}
    meta['stream']=np.char.add(np.char.add(meta['subject'],'|'),meta['session'])
    person=np.array([s.split('/')[-1] for s in meta['subject']])
    frozen,_=ref.controls_and_age(meta)
    allbanks={}
    proof={}
    for seed in (379,1441):
        root=Path(str(seed))
        matched=Path('..')/('ni26_gain439_20261118' if seed==379 else 'ni26_gain441_20261118')/'coherent'
        banks=dict(frozen)
        for label,folder in [('anchor',root),('free',matched)]:
            predictions,proofs=ref.folds(folder,ref.source_manifest(folder/'source.sha256'),person)
            banks.update({label+'_'+k:v for k,v in predictions.items()})
            proof[f'{seed}|{label}']=proofs
        for p in people:
            check=json.loads((root/(p+'.parity.json')).read_text())
            assert check['seed']==seed and check['held']==p and check['static_norm'] and check['max_anchor_error']<2e-5
            assert check['matched']==proof[f'{seed}|free'][p]['archive_sha256']
        allbanks[seed]=banks
    Path('before.json').write_text(json.dumps(proof,indent=2))
    with np.load(data) as z:
        y=z['y']
    sys.path[:0]=[str(ref.METRICS),str(ref.BASE/'studies/14nov/newmember')]
    assert ref.sha(ref.METRICS/'source.sha256')=='9e889431c92f4759c79e0724446d19fc9503ac5b726464f73f9ed22b548f9ccd'
    ref.source_manifest(ref.METRICS/'source.sha256')
    assert ref.sha(ref.BASE/'studies/14nov/newmember/score_s_v66p.py')=='ae387cd5a7b116fc1259001d4fe449ba1065b0cb5943c117d4f65ec87a0b40d4'
    from measure import official
    from score_s_v66p import cell_weighted_summary,summarize
    metric=official()
    cells=np.char.add(np.char.add(meta['stream'],'|'),meta['interface'])
    counts=np.array([len(set(cells[person==p])) for p in people])
    report={}
    for seed,banks in allbanks.items():
        tables={k:np.array([[100*metric(y[person==p],row[person==p],meta,person==p)['balanced_accuracy_cell_mean']
                             for p in people] for row in pred]) for k,pred in banks.items()}
        scores={k:float(np.average((v[:4].mean(0)+v[4:6].mean(0)+v[6:].mean(0))/3,weights=counts))
                for k,v in tables.items()}
        assert abs(scores['frozen']-76.27933502146)<1e-7 and abs(scores['JS']-76.55173151211)<1e-7
        assert abs(scores['free_memory']-({379:43.92346736386,1441:45.62918214344}[seed]))<1e-7
        contrasts={}
        for other in ('free_memory','anchor_blank','anchor_donor','anchor_norm','anchor_static'):
            delta=tables['anchor_memory']-tables[other]
            contrasts[other]=dict(official=cell_weighted_summary(delta,rows,counts),equal_people=summarize(delta,rows,people))
        report[seed]=dict(scores=scores,contrasts=contrasts)
    Path('result.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
