"""Fixed scale contrasts and equal-cell per-task attribution from committed decisions."""
from pathlib import Path
import json
import numpy as np
from fit import ARMS
import score379 as base

result=json.loads(Path('result391.json').read_text())['all']
with np.load(base.BASE/'data/proteus_windows.npz') as z:
    people=np.array([s.split('/')[-1] for s in z['subject']])
    cell=np.char.add(np.char.add(np.char.add(people,'|'),z['session']),z['interface'])
    y=z['y']
base.ARMS=ARMS
banks,_=base.folds(Path.cwd(),base.source_manifest(Path('source.sha256')),people)
class_diff=[]
for k in range(3):
    recalls=[]
    for label in np.unique(cell):
        mask=(cell==label)&(y==k)
        correct=(banks['transition'][:,mask]==k).astype(float)-(banks['scrambled'][:,mask]==k)
        recalls.append(100*correct.mean())
    class_diff.append(float(np.mean(recalls)))
assert np.isclose(np.mean(class_diff),result['scores']['transition']-result['scores']['scrambled'])
out=dict(task_T_minus_D=class_diff,delta={s:result['contrasts'][s+'-marginal'] for s in ('d1','d3','d8')})
Path('scale_task.json').write_text(json.dumps(out,indent=2)+'\n')
