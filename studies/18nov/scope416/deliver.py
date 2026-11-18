"""Deliver complete session streams with the immutable §385 worker."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
from worker import run, replay

arm, person = sys.argv[1], int(sys.argv[2])
assert torch.__version__ == '2.14.1+cu130' and 'A100' in torch.cuda.get_device_name()
with np.load('/external.npz') as z:
    x = torch.from_numpy(z['X'])
    people, days, names = z['subject'], z['session'], z['ch_names'].tolist()
assert x.shape == (2130, 41, 480) and x.dtype == torch.float32
subject = sorted(set(people))[max(0, person)]
model = replay.load(Path('/package'), names)
result = {}
with np.load('orders.npz') as bank:
    for key in ('fixed411', 'moved411', 'fixed1411', 'moved1411'):
        for day in sorted(set(days[people == subject])):
            rows = bank[key][(people == subject) & (days == day)]
            if person == -1:
                rows = rows[:35]
            rec = run(model, x, rows, arm)
            result[f'{subject}|{day}|{key}'] = rec
            if person == -1:
                assert run(model, x, rows[:17], arm)['labels'] == rec['labels'][:17]
                break
if person == -1 and arm == 'cap':
    px, ps, pn = replay.read_source()
    assert pn == names
    control = run(model, px, ps[0][1][:31], arm)
    prior = json.loads(Path('/prior.json').read_text())['streams']['first']['P01-1']
    assert control['labels'] == prior['labels'][:31]
assert 'neural-interfaces-2026/.venv/' not in Path('/proc/self/maps').read_text()
Path(f'{arm}_{person}.json').write_text(json.dumps(result))
