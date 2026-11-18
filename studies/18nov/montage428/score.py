"""Compare matched real package decisions, resampling people, days and orders."""
from pathlib import Path
import json
import numpy as np


def read(path):
    return json.loads(path.read_text())


def table(path, y):
    expected = read(path/'result.json')
    cells = np.empty((9,2,2,2))
    decisions = {}
    for i, person in enumerate(expected['people']):
        for a, arm in enumerate(expected['arms']):
            rec = read(path/f'{arm}_{i}.json')
            assert len(rec) == 4
            for name, r in rec.items():
                subject, day, seed = name.split('|')
                assert person == subject
                day, order = ('1','2').index(day), ('411','1411').index(seed)
                labels = np.array(r['labels'])
                rows = np.array(r['rows'])
                vals = y[rows]
                cells[i,day,order,a] = 100*np.mean([(labels[vals==c]==c).mean() for c in range(3)])
                decisions[(arm,name)] = (rows,labels)
    np.testing.assert_allclose(cells.mean((0,1,2)),expected['acc'])
    return cells, decisions


def paired(values):
    rng = np.random.default_rng(428)
    n = len(values)
    people = rng.integers(n,size=(10000,n,1,1))
    days = rng.integers(2,size=(10000,n,2,1))
    orders = rng.integers(2,size=(10000,n,2,2))
    return dict(mean=float(values.mean()), ci80=np.quantile(
        values[people,days,orders].mean((1,2,3)), [.1,.9]).tolist())


with np.load('local.npz') as z:
    labels = z['y']
base, prev = table(Path('../ni26_cross411b_20261118'), labels)
alt, curr = table(Path('lane'), labels)
changed = {arm:0 for arm in ('old','cap')}
assert set(prev) == set(curr)
for key,(rows,pred) in prev.items():
    np.testing.assert_array_equal(rows, curr[key][0])
    changed[key[0]] += int(np.sum(pred != curr[key][1]))
delta = (alt-base)[...,1]-(alt-base)[...,0]
records = [read(Path(f'p{i}.json')) for i in range(9)]
rms = np.array([d['held_rms_uv'] for d in records]).mean(2)
summary = {}
for name, idx in [('primary',slice(0,None,2)), ('reserve',slice(1,None,2)), ('all',slice(None))]:
    summary[name] = dict(interaction=paired(delta[idx]),
        stock=base[idx].mean((0,1,2)).tolist(), local=alt[idx].mean((0,1,2)).tolist(),
        holdout_rms_uv=rms[idx].mean(0).tolist())
out = dict(cohorts=summary,changed=changed,person_interaction=delta.mean((1,2)).tolist(),
    person_holdout_rms_uv=rms.tolist(),maximum_baseline_error_uv=max(d['max_error_uv'] for d in records),
    source_channels=records[0]['held'])
Path('result428.json').write_text(json.dumps(out,indent=2)+'\n')
