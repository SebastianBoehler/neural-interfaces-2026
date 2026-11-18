"""Time official predictions; collect posterior bytes only in parity runs."""
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
import replay
from benchmark_utils.streaming import fresh_model


def stream(template, bank, rows, observer):
    torch.cuda.synchronize()
    start = perf_counter()
    model = fresh_model(template)
    captures = []
    if observer:
        predict_cp = model.cp.step

        def capture(*args):
            posterior = predict_cp(*args)
            captures.append(posterior.copy())
            return posterior

        model.cp.step = capture
    labels = []
    for row in rows:
        label = model.predict(bank[row:row + 1].to('cuda:0'))
        assert label.device.type == 'cpu' and label.dtype == torch.int64 and label.shape == (1,)
        labels.append(int(label[0]))
    torch.cuda.synchronize()
    elapsed = perf_counter() - start
    activity = replay.activity(model)
    assert activity['windows'] == len(rows) and not activity['errors'] and not activity['ft_over']
    assert not model.dead and not model.cp.dead and not model.cf.dead
    output = dict(seconds=elapsed, labels=labels, activity=activity, latch=model.cf.latch,
                  views=len(model.cf.V), rows_sha=hashlib.sha256(rows.astype('<i8').tobytes()).hexdigest())
    if observer:
        probs = np.stack(captures)
        assert probs.shape == (len(rows), 3) and np.isfinite(probs).all()
        assert probs.argmax(1).tolist() == labels
        output.update(posterior_sha=hashlib.sha256(probs.tobytes()).hexdigest(), dtype=probs.dtype.str)
    return output


def main(arm, mode, replica):
    assert torch.__version__ == '2.14.1+cu130' and 'A100' in torch.cuda.get_device_name()
    if mode in ('smoke', 'parity'):
        torch.manual_seed(408)
        np.random.seed(408)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    px, sessions, channels = replay.read_source()
    known = dict(sessions)
    with np.load('/external.npz') as z:
        assert z['ch_names'].tolist() == channels and not len(z['interpolated'])
        ext = torch.from_numpy(z['X'])
        p, d, r = z['subject'], z['session'], z['run']
    tasks = [('P10-3', px, known['P10-3'])]
    for i, length in ((1, 320), (2, 813)):
        name = f'Schirrmeister2017/{i}'
        rows = np.flatnonzero((p == name) & (d == '0') & (r == '0train'))
        assert len(rows) == length
        tasks.append((name, ext, np.random.default_rng([387, i]).permutation(rows)))
    prior = json.loads(Path('/prior.json').read_text())['streams']
    model = replay.load(Path('/package'), channels)
    stream(model, px, known['P01-1'][:31], False)
    warm = stream(model, ext, tasks[1][2][:35], False)
    assert warm['activity']['ft_steps'] > 0
    records = {}
    for name, bank, rows in tasks:
        if mode in ('smoke', 'diagnose'):
            rows = rows[:65]
        record = stream(model, bank, rows, mode != 'time')
        assert record['labels'] == prior[name]['labels'][:len(rows)]
        a = record['activity']
        assert (a['recognized'] >= 0) == (name == 'P10-3')
        assert (a['ft_steps'] > 0) == (name != 'P10-3')
        if mode in ('time', 'parity'):
            assert a == prior[name]['activity']
        records[name] = record
    Path(f'{arm}-{replica}.json').write_text(json.dumps(records))


if __name__ == '__main__':
    main(*sys.argv[1:])
