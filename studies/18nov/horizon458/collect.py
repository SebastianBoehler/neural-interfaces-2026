"""Authenticate blind decisions and compare official cells under paired resampling."""
import json
from pathlib import Path

import numpy as np
from fusion_inputs import official
from score import interval
from stocks456 import inputs,ref


def collect():
    source=ref.sha(Path('source.sha256'))
    banks={}
    for p in ref.PEOPLE:
        path=Path(p+'.npz')
        receipt=json.loads(Path(p+'.json').read_text())
        assert receipt==dict(bank=ref.sha(path),source=source)
        with np.load(path) as archive:
            banks.update(dict(archive))
    data=ref.BASE/'data/proteus_windows.npz'
    assert ref.sha(data)==ref.DATA_SHA
    with np.load(data) as z:
        labels=z['y']
    meta=inputs()[0]
    metric=official()
    scores={a:np.empty((10,8)) for a in ('stock','454','1454','uniform','mixed','recent','mixed_donor')}
    counts=[]
    for i,p in enumerate(ref.PEOPLE):
        mask=meta['person']==p
        counts.append(len(set(zip(meta['stream'][mask],meta['interface'][mask]))))
        for j,o in enumerate(ref.ORDERS):
            for a in scores:
                res=metric(labels[mask],banks['|'.join((p,o,a))],meta,mask)
                assert res['n_cells']==counts[-1]
                scores[a][i,j]=100*res['balanced_accuracy_cell_mean']
    counts=np.array(counts)
    estimates={a:interval(v,counts)['pp'] for a,v in scores.items()}
    assert abs(estimates['uniform']-76.28077827233042)<1e-9
    contrasts={a:interval(scores['mixed']-v,counts) for a,v in scores.items() if a!='mixed'}
    contrasts['recent_minus_uniform']=interval(scores['recent']-scores['uniform'],counts)
    return dict(source=source,banks={p:ref.sha(Path(p+'.npz')) for p in ref.PEOPLE},
                absolute=estimates,contrasts=contrasts)


if __name__=='__main__':
    Path('result.json').write_text(json.dumps(collect(),indent=2))
