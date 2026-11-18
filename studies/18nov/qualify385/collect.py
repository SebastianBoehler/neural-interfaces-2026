"""Paired external balanced accuracy after the complete candidate banks pass."""
from pathlib import Path
import json
import sys
import numpy as np
from data import permutation
from protocol import score

root = Path('/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026')
with np.load(root / 'data/scherer2015/windows.npz') as z:
    people, days, interface = (z[k] for k in ('subject', 'session', 'interface'))
streams = np.char.add(np.char.add(people, '|'), days)
orders = [permutation(streams, None, 'full', s) for s in (0, 1)]
arms = ('old', 'cap')
pred = np.full((2, 2, len(people)), -1, int)
activity = {}
for i, subject in enumerate(sorted(set(people))):
    for a, arm in enumerate(arms):
        records = json.loads(Path(f'{arm}_{i}.json').read_text())
        assert len(records) == 4
        for name, rec in records.items():
            p, day, s = name.split('|')
            s = int(s)
            expected = orders[s][(people == p) & (days == day)]
            assert p == subject and np.array_equal(expected, rec['rows'])
            assert (pred[a,s,expected] == -1).all() and len(expected) == len(rec['labels'])
            pred[a,s,expected] = rec['labels']
            activity[f'{arm}|{name}'] = rec['activity']
assert np.isin(pred, (0, 1, 2)).all() and len(activity) == 72
with np.load(root / 'data/scherer2015/windows.npz') as z:
    y = z['y']
cells = np.empty((9, 2, 2, 2, 3))
for i, person in enumerate(sorted(set(people))):
    for j, day in enumerate(sorted(set(days[people == person]))):
        for c in range(3):
            mask = (people == person) & (days == day) & (y == c)
            assert mask.any()
            cells[i,j,:,:,c] = 100*(pred[:,:,mask] == c).mean(-1).T
for a in range(2):
    for s in range(2):
        check = score(y, pred[a,s], dict(stream=streams, interface=interface), np.ones(len(y), bool))
        assert check['n_cells'] == 18 and np.isclose(100*check['balanced_accuracy_cell_mean'], cells[:,:,s,a,:].mean())
diff = (cells[:,:,:,1,:]-cells[:,:,:,0,:]).mean(-1)
rng = np.random.default_rng(385)
b = 10000
i = rng.integers(9, size=(b,9,1,1))
j = rng.integers(2, size=(b,9,2,1))
k = rng.integers(2, size=(b,9,2,2))
ci = np.quantile(diff[i,j,k].mean((1,2,3)), [.1,.9]).tolist()
res = dict(arms=arms, people=sorted(set(people)), acc=cells.mean((0,1,2,4)).tolist(),
           classes=cells.mean((0,1,2)).tolist(), diff=float(diff.mean()), ci80=ci,
           order_diff=diff.mean((0,1)).tolist(), person_diff=diff.mean((1,2)).tolist(),
           changed=int((pred[0]!=pred[1]).sum()), activity=activity)
Path('result.json').write_text(json.dumps(res, indent=2)+'\n')
print({k:v for k,v in res.items() if k != 'activity'})
