"""Warm paired native replay on complete sessions, changing only optimizer work."""
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
import replay
from data import permutation
from benchmark_utils.streaming import fresh_model

ARMS = ('stock', 'lower', 'zero', 'alternate')


def stream(template, x, rows, arm, seed):
    torch.cuda.synchronize()
    start = perf_counter()
    model = fresh_model(template)
    ft = model.ft
    assert ft.on and ft.n_steps == 5 and len(ft.member) == 4
    labels = []
    for row in rows:
        if arm == 'alternate':
            ft.n_steps = 4 if (ft.refits + seed) % 2 == 0 else 0
        elif arm != 'stock':
            ft.n_steps = 2 if arm == 'lower' else 0
        label = model.predict(x[row:row + 1].to('cuda:0'))
        assert label.device.type == 'cpu' and label.dtype == torch.int64 and label.shape == (1,)
        labels.append(int(label[0]))
    torch.cuda.synchronize()
    activity = replay.activity(model)
    assert not model.dead and activity['errors'] == 0
    del ft, model
    torch.cuda.synchronize()
    return dict(labels=labels, activity=activity, seconds=perf_counter()-start)


def main(index):
    assert 'A100' in torch.cuda.get_device_name() and torch.__version__ == '2.14.1+cu130'
    with np.load('/external.npz') as archive:
        people, days, names = (archive[k] for k in ('subject', 'session', 'ch_names'))
        x = torch.from_numpy(archive['X'])
    assert x.shape == (2130, 41, 480) and x.dtype == torch.float32
    person = sorted(set(people))[max(0, index)]
    references = json.loads(Path(f'/prior/cap_{max(0,index)}.json').read_text())
    sessions = np.char.add(np.char.add(people, '|'), days)
    template = replay.load(Path('/package'), names.tolist())
    result = {}
    for seed in (0, 1):
        order = permutation(sessions, None, 'full', seed)
        for day_id, day in enumerate(sorted(set(days[people == person]))):
            key = f'{person}|{day}|{seed}'
            rows = order[(people == person) & (days == day)]
            ref = references[key]
            assert rows.tolist() == ref['rows']
            if not result:
                warm = stream(template, x, rows[:31], 'stock', seed)
                assert warm['labels'] == ref['labels'][:31] and warm['activity']['ft_refits'] > 0
            if index < 0:
                rows = rows[:41]
            offset = (2*seed + day_id + max(0, index)) % 4
            rec = result[key] = {'rows': rows.tolist(), 'arms': {}}
            for arm in ARMS[offset:] + ARMS[:offset]:
                run = rec['arms'][arm] = stream(template, x, rows, arm, seed)
                assert run['labels'][:20] == ref['labels'][:20]
                activity = run['activity']
                n = activity['ft_refits']
                assert n == activity['ft_relabels'] and n > 0
                steps = {'stock': 20*n, 'lower': 8*n, 'zero': 0, 'alternate': 16*((n+1-seed)//2)}
                assert activity['ft_steps'] == steps[arm]
                assert set(run['labels']) <= {0, 1, 2}
                if index >= 0:
                    expected = dict(ref['activity'], ft_steps=steps[arm])
                    assert activity == expected, (key, arm, activity, expected)
                if arm == 'stock':
                    assert run['labels'] == ref['labels'][:len(rows)]
            if index < 0:
                break
        if index < 0:
            break
    assert 'neural-interfaces-2026/.venv/' not in Path('/proc/self/maps').read_text()
    Path(f'person_{index}.json').write_text(json.dumps(result))


if __name__ == '__main__':
    main(int(sys.argv[1]))
