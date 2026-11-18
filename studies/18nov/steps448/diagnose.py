"""Retain the failing stock row and distinguish cold, warm and after-intervention replay."""
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

import replay
from worker import stream


def main():
    with np.load('/external.npz') as archive:
        x = torch.from_numpy(archive['X'])
        channels = archive['ch_names'].tolist()
    person = int(sys.argv[1])
    ref = json.loads(Path(f'/prior/cap_{person}.json').read_text())
    key, reference = next(iter(ref.items()))
    rows = np.array(reference['rows'])
    expected = np.array(reference['labels'])
    model = replay.load(Path('/package'), channels)
    output = dict(key=key, gpu=torch.cuda.get_device_name(), cpu=os.uname().machine, runs={})
    for name, arm, length in [('cold', 'stock', len(rows)), ('warmup', 'stock', 31),
                              ('lower', 'lower', len(rows)), ('zero', 'zero', len(rows)),
                              ('alternate', 'alternate', len(rows)), ('after', 'stock', len(rows))]:
        run = stream(model, x, rows[:length], arm, 0)
        wrong = np.flatnonzero(np.array(run['labels']) != expected[:length])
        output['runs'][name] = dict(activity=run['activity'], different=[
            [int(i), int(rows[i]), int(expected[i]), run['labels'][i]] for i in wrong])
        Path(f'diagnosis_{person}.json').write_text(json.dumps(output))
    if output['runs']['cold']['different'] == output['runs']['after']['different']:
        print('identical stock decisions across cold and after-intervention streams')


if __name__ == '__main__':
    main()
