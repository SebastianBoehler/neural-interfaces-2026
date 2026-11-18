"""Compare each frozen archive to its earlier run, including decision-level flips."""
from pathlib import Path
import json
import numpy as np

baseline = Path('../ni26_cross411b_20261118')
metric = [json.loads((folder/'result.json').read_text()) for folder in (baseline, Path('.'))]
details = {}
with np.load('../neural-interfaces-2026/data/scherer2015/windows.npz') as z:
    labels = z['y']
for arm in ('old', 'cap'):
    flips, counter_changes, delivered = [], [], 0
    for person in range(9):
        filename = f'{arm}_{person}.json'
        old, new = [json.loads((folder/filename).read_text()) for folder in (baseline, Path('.'))]
        assert old.keys() == new.keys() and len(old)==4
        for key, a in old.items():
            b = new[key]
            assert a['rows'] == b['rows'] and len(a['rows']) == len(b['labels']) == len(a['labels'])
            if a['activity'] != b['activity']:
                counter_changes.append(dict(stream=key, old=a['activity'], new=b['activity']))
            for index in np.flatnonzero(np.array(a['labels']) != b['labels']):
                row = a['rows'][index]
                flips.append(dict(stream=key, position=int(index)+1, row=row, truth=int(labels[row]),
                                  original=a['labels'][index], repeat=b['labels'][index]))
            delivered += len(a['rows'])
    assert delivered == 4260
    details[arm] = dict(delivered=delivered, flips=flips, counter_changes=counter_changes)
out = dict(original=metric[0]['acc'], repeat=metric[1]['acc'], original_delta=metric[0]['diff'],
           repeat_delta=metric[1]['diff'], delta_shift=metric[1]['diff']-metric[0]['diff'], details=details)
Path('comparison.json').write_text(json.dumps(out,indent=2)+'\n')
