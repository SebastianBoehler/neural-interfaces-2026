"""Paired process/session timing with exact archived prefix guards."""
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
import replay
from benchmark_utils.streaming import fresh_model


def synchronized_time():
    torch.cuda.synchronize()
    return perf_counter()


def session(template, eeg, rows, arm, labels, returning):
    start = synchronized_time()
    model = fresh_model(template)
    copied = synchronized_time()
    predictions = []
    for row in rows:
        label = model.predict(eeg[row:row+1].to('cuda:0'))
        assert label.device.type == 'cpu' and label.dtype == torch.int64 and label.shape == (1,)
        predictions.append(int(label[0]))
    predicted = synchronized_time()
    assert predictions == labels and set(labels) <= {0, 1, 2}
    if arm == 'cap':
        activity = replay.activity(model)
        assert activity['errors'] == 0 and activity['windows'] == len(rows) and not model.dead
        assert (activity['recognized'] >= 0) == returning
        assert (activity['ft_steps'] > 0) != returning
    else:
        activity = dict(model.route.log)
        assert activity['err'] == 0 and activity['n'] == len(rows)
    del model
    ended = synchronized_time()
    return dict(copy=copied-start, predict=predicted-copied, total=ended-start, activity=activity)


def main(arm, rep):
    assert 'A100' in torch.cuda.get_device_name() and torch.__version__ == '2.14.1+cu130'
    x, public, channels = replay.read_source()
    with np.load('/external.npz') as data:
        assert data['ch_names'].tolist() == channels
        ext = torch.from_numpy(data['X'])
        source, day, run = data['subject'], data['session'], data['run']
    names = ['P10-3', 'Schirrmeister2017/1', 'Schirrmeister2017/2', 'P10-3']
    if rep == 1:
        names[1:3] = names[1:3][::-1]
    reference = json.loads(Path(f'/prior/{arm}_0.json').read_text())['streams']
    started = synchronized_time()
    template = replay.load(Path('/package'), channels)
    loaded = synchronized_time()
    streams = []
    for name in names:
        returning = name == 'P10-3'
        if returning:
            rows = dict(public)[name]
        else:
            indices = np.flatnonzero((source == name) & (day == '0') & (run == '0train'))
            rows = np.random.default_rng([387, int(name[-1])]).permutation(indices)
        assert rows.tolist() == reference[name]['rows']
        rows = rows[:35 if rep == -1 else 190]
        streams.append(session(template, x if returning else ext, rows, arm,
                               reference[name]['labels'][:len(rows)], returning))
    Path(f'{arm}_{rep}.json').write_text(json.dumps(dict(
        names=names, load=loaded-started, streams=streams, gpu=torch.cuda.get_device_name())))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
