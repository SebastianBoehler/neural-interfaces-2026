"""Check which classes the complete, paired decoder decisions recover or lose."""
import json
from pathlib import Path

import numpy as np
from score379 import BASE, CONTROL, ORDERS, controls_and_age, folds, sha, source_manifest

root = Path.cwd()
primary = json.loads((root/'result379.json').read_text())
with np.load(BASE/'data/proteus_windows.npz') as z:
    y, subjects, sessions, interfaces = [z[k] for k in ('y','subject','session','interface')]
people = np.array([s.split('/')[-1] for s in subjects])
stream = np.char.add(np.char.add(subjects,'|'),sessions)
frozen,_ = controls_and_age(dict(subject=subjects,stream=stream))
bank,_ = folds(root,source_manifest(root/'source.sha256'),people)
bank.update(frozen)
cells = np.char.add(np.char.add(stream,'|'),interfaces)
results = {}
for arm,pred in bank.items():
    records, missing = [], np.zeros(3,dtype=int)
    for cell in np.unique(cells):
        active = cells == cell
        recalls = []
        for label in range(3):
            rows = active & (y == label)
            assert rows.any()
            correct = (pred[:,rows] == label).mean(1)
            recalls.append(np.mean([correct[:4].mean(),correct[4:6].mean(),correct[6:].mean()]))
            missing[label] += (pred[:,active] != label).all(1).sum()
        records.append(recalls)
    results[arm] = dict(recall_pct=(100*np.mean(records,0)).tolist(),no_prediction_cell_orders=missing.tolist())
Path('classes379.json').write_text(json.dumps(dict(primary_sha256=sha(root/'result379.json'),
    cells=len(set(cells)),orders=ORDERS,classes_by_label=results),indent=2))
