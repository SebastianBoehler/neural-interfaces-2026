"""Reuse the exact F-C4 optimizer hook on §411b delivery seeds."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

from balanced_actor import instrument
from data import permutation, sha
import external_actor as worker
import replay

ORDERS = (411,1411)


def run(index, arm, smoke):
    assert arm in ('stock','primary')
    assert torch.__version__=='2.14.1+cu130' and 'A100' in torch.cuda.get_device_name()
    worker.instrument=instrument
    with np.load('/external.npz',allow_pickle=False) as z:
        x=torch.from_numpy(z['X'])
        people,days,channels=(z[k] for k in ('subject','session','ch_names'))
    assert x.shape==(2130,41,480) and x.dtype==torch.float32
    person=sorted(set(people))[index]
    template=replay.load(Path('/package'),channels.tolist())
    groups=np.char.add(np.char.add(people,'|'),days)
    archive=json.loads(Path(f'/prior/cap_{index}.json').read_text())
    records={}
    for order in ORDERS:
        perm=permutation(groups,None,'full',order)
        for day in sorted(set(days[people==person])):
            name=f'{person}|{day}|{order}'
            rows=perm[(people==person)&(days==day)]
            anchor=archive[name]
            assert rows.tolist()==anchor['rows']
            out=worker.stream(template,x,rows,arm)
            assert out['activity']==anchor['activity']
            assert out['labels'][:20]==anchor['labels'][:20]
            if arm=='stock':
                assert out['labels']==anchor['labels']
            events=out['events']
            assert events and any(e['unequal'] for e in events)
            if arm=='stock':
                assert all(e['changed']==0 for e in events)
            else:
                assert all(e['changed']>0 for e in events if e['unequal'])
            records[name]=dict(rows=rows.tolist(),**out)
            if smoke:
                prefix=worker.stream(template,x,rows[:61],arm)
                assert prefix['labels']==out['labels'][:61]
                assert [e['age'] for e in prefix['events']]==[e['age'] for e in events][:5]
                break
        if smoke:
            break
    assert len(records)==(1 if smoke else 4)
    assert 'neural-interfaces-2026/.venv/' not in Path('/proc/self/maps').read_text()
    path=Path(f'{arm}_{index}{"_smoke" if smoke else ""}.json')
    path.write_text(json.dumps(dict(source=sha(Path('source.sha256')),sessions=records))+'\n')
    path.with_suffix('.sha256').write_text(sha(path)+'\n')
    print(person,arm,len(records),'full exact sessions verified',flush=True)


if __name__=='__main__':
    run(int(sys.argv[1]),sys.argv[2],sys.argv[3:]==['smoke'])
