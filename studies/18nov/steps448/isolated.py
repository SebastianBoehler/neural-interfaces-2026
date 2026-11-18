"""One canonical accuracy sequence and a warm first-stream timing per clean process."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

import replay
from data import permutation
from worker import stream


def main(index, arm):
    assert torch.__version__ == '2.14.1+cu130' and 'A100' in torch.cuda.get_device_name()
    i = 5 if index < 0 else index
    controls = json.loads(Path(f'/prior/cap_{i}.json').read_text())
    with np.load('/external.npz') as archive:
        x = torch.from_numpy(archive['X'])
        people, days, names = (archive[k] for k in ('subject', 'session', 'ch_names'))
    person = sorted(set(people))[i]
    sessions = np.char.add(np.char.add(people, '|'), days)
    template = replay.load(Path('/package'), names.tolist())
    output = {}
    for seed in (0, 1):
        order = permutation(sessions, None, 'full', seed)
        for day in sorted(set(days[people == person])):
            key = f'{person}|{day}|{seed}'
            ref = controls[key]
            rows = order[(people == person) & (days == day)]
            assert ref['rows'] == rows.tolist()
            if index < 0:
                rows = rows[:41]
            rec = stream(template, x, rows, arm, seed)
            n = rec['activity']['ft_refits']
            steps = {'stock': 20*n, 'lower': 8*n, 'zero': 0, 'alternate': 16*((n+1-seed)//2)}
            assert rec['activity']['ft_steps'] == steps[arm]
            if index >= 0:
                assert rec['activity'] == dict(ref['activity'], ft_steps=steps[arm])
            assert rec['labels'][:20] == ref['labels'][:20]
            if arm == 'stock':
                assert rec['labels'] == ref['labels'][:len(rows)], key
            rec['rows'] = rows.tolist()
            output[key] = rec
            if index < 0:
                break
        if index < 0:
            break
    first = next(iter(output.values()))
    again = stream(template, x, np.array(first['rows']), arm, 0)
    assert again['activity'] == first['activity']
    first['cold_seconds'], first['seconds'] = first['seconds'], again['seconds']
    first['warm_differences'] = np.flatnonzero(np.array(first['labels']) != again['labels']).tolist()
    Path(f'{arm}_{index}.json').write_text(json.dumps(output))


if __name__ == '__main__':
    main(int(sys.argv[1]), sys.argv[2])
