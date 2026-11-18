"""Validate blind deliveries, then score paired person/day/order effects."""
import json
from pathlib import Path
import numpy as np
from protocol import score

full = Path('../ni26_cross411b_20261118')
path = Path('../neural-interfaces-2026/data/scherer2015/windows.npz')
with np.load(path) as z:
    people, days, interface = [z[k] for k in ('subject','session','interface')]
streams = np.char.add(np.char.add(people, '|'), days)
subjects = sorted(set(people))
kinds = ('full', 'fixed', 'moved')
seeds = (411, 1411)
arms = ('old', 'cap')
pred = np.full((3,2,2,len(people)), -1)
activity = {}
with np.load('orders.npz') as orders:
    for i, subject in enumerate(subjects):
        for a, arm in enumerate(arms):
            for bank, kinds_here in [(full, ('full',)), (Path('.'), ('fixed','moved'))]:
                records = json.loads((bank/f'{arm}_{i}.json').read_text())
                assert len(records) == len(kinds_here)*4
                for (key, rec) in records.items():
                    person, day, label = key.split('|')
                    kind = next(k for k in kinds_here if k == 'full' or label.startswith(k))
                    seed = int(label.removeprefix(kind))
                    idx = kinds.index(kind), seeds.index(seed), a
                    rows = orders[f'{kind}{seed}'][(people == subject) & (days == day)]
                    assert person == subject and np.array_equal(rows, rec['rows'])
                    assert np.all(pred[idx][rows] == -1) and len(rows) == len(rec['labels'])
                    pred[idx][rows] = rec['labels']
                    act = rec['activity']
                    assert act['err' if a == 0 else 'errors'] == 0
                    assert act['n' if a == 0 else 'windows'] == len(rows)
                    activity[f'{kind}|{arm}|{person}|{day}|{seed}'] = act
assert np.isin(pred, (0,1,2)).all() and len(activity) == 216
with np.load(path) as z:
    y = z['y']
cells = np.empty((9,2,2,3,2))  # person, day, seed, scheme, arm
for i, person in enumerate(subjects):
    for j, day in enumerate(sorted(set(days[people == person]))):
        subset = (people == person) & (days == day)
        cells[i,j] = 100*np.mean([(pred[..., subset & (y == c)] == c).mean(-1)
                                 for c in range(3)], axis=0).transpose(1,0,2)
means = cells.mean((0,1))
for k in range(3):
    for s in range(2):
        for a in range(2):
            out = score(y,pred[k,s,a],dict(stream=streams,interface=interface),np.ones(len(y),bool))
            assert out['n_cells'] == 18 and np.isclose(100*out['balanced_accuracy_cell_mean'],means[s,k,a])
prior = json.loads((full/'result.json').read_text())
assert np.allclose(means[:,0,:].mean(0),prior['acc'],rtol=0,atol=1e-10)
delta = cells[...,1] - cells[...,0]
rng = np.random.default_rng(416)
p = rng.integers(9, size=(10000,9,1,1))
d = rng.integers(2, size=(10000,9,2,1))
s = rng.integers(2, size=(10000,9,2,2))

def describe(x):
    boot = x[p,d,s].mean((1,2,3))
    return dict(mean=x.mean(), ci80=np.quantile(boot,[.1,.9]).tolist(),
                people=x.mean((1,2)).tolist(), orders=x.mean((0,1)).tolist())

res = dict(people=subjects, scores={k:dict(zip(arms,means[:,i].mean(0))) for i,k in enumerate(kinds)},
           delta={k:describe(delta[...,i]) for i,k in enumerate(kinds)},
           blocked_minus_full=describe(delta[...,1:].mean(-1)-delta[...,0]),
           moved_minus_fixed=describe(delta[...,2]-delta[...,1]))
Path('result.json').write_text(json.dumps(res,indent=2)+'\n')
Path('activity.json').write_text(json.dumps(activity,sort_keys=True)+'\n')
print(res)
