"""Predict from immutable ZIP members, using official worker session boundaries."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
import replay
from data import permutation
from benchmark_utils.streaming import fresh_model


def run(template, x, rows, arm):
    model = fresh_model(template)
    labels = []
    for row in rows:
        label = model.predict(x[int(row):int(row)+1].to('cuda:0'))
        assert label.device.type == 'cpu' and label.shape == (1,) and label.dtype == torch.int64
        labels.append(int(label[0]))
    assert set(labels) <= {0, 1, 2}
    if arm == 'cap':
        act = replay.activity(model)
        assert act['windows'] == len(rows) and act['errors'] == 0 and not model.dead
    else:
        act = dict(model.route.log)
        assert act['n'] == len(rows) and act['err'] == 0
    return dict(rows=rows.tolist(), labels=labels, activity=act)


def main(arm, person):
    assert torch.__version__ == '2.14.1+cu130' and 'A100' in torch.cuda.get_device_name()
    with np.load('/external.npz') as z:
        x = torch.from_numpy(z['X'])
        people, days, names = z['subject'], z['session'], z['ch_names'].tolist()
    assert x.shape == (2130, 41, 480) and x.dtype == torch.float32
    streams = np.char.add(np.char.add(people, '|'), days)
    subject = sorted(set(people))[max(person, 0)]
    model = replay.load(Path('/package'), names)
    result = {}
    for seed in (0, 1):
        order = permutation(streams, None, 'full', seed)
        for day in sorted(set(days[people == subject])):
            rows = order[(people == subject) & (days == day)]
            if person == -1:
                rows = rows[:96]
            rec = run(model, x, rows, arm)
            result[f'{subject}|{day}|{seed}'] = rec
            if person == -1:
                assert run(model, x, rows[:17], arm)['labels'] == rec['labels'][:17]
                if arm == 'cap':
                    px, ps, pn = replay.read_source()
                    assert pn == names
                    control = run(model, px, ps[0][1][:31], arm)
                    prev = json.loads(Path('/prior.json').read_text())['streams']['first']['P01-1']
                    assert control['labels'] == prev['labels'][:31]
                break
        if person == -1:
            break
    assert 'neural-interfaces-2026/.venv/' not in Path('/proc/self/maps').read_text()
    Path(f'{arm}_{person}.json').write_text(json.dumps(result))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
