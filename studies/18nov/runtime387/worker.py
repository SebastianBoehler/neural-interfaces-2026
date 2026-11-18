"""Measure complete exact-archive sessions with natural routing and paired rows."""
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
import replay
from worker385 import run


def rows_hash(rows):
    return hashlib.sha256(rows.astype('<i8').tobytes()).hexdigest()


def main(arm, mode, output):
    assert torch.__version__ == '2.14.1+cu130' and 'A100' in torch.cuda.get_device_name()
    x, public, channels = replay.read_source()
    public = dict(public)
    p10 = public['P10-3']
    assert len(p10) == max(map(len, public.values())) == 199
    with np.load('/external.npz') as z:
        assert z['ch_names'].tolist() == channels and not len(z['interpolated'])
        ext = torch.from_numpy(z['X'])
        p, d, r = z['subject'], z['session'], z['run']
    jobs = [('P10-3', x, p10)]
    for i, length in [(1,320),(2,813)]:
        rows = np.flatnonzero((p == f'Schirrmeister2017/{i}') & (d == '0') & (r == '0train'))
        assert rows.size == length
        jobs.append((f'Schirrmeister2017/{i}', ext, np.random.default_rng([387,i]).permutation(rows)))
    prior = json.loads(Path('/prior.json').read_text())['streams']['first']
    model = replay.load(Path('/package'), channels)
    warm = run(model, x, public['P01-1'][:31], arm)
    if arm == 'cap':
        assert warm['labels'] == prior['P01-1']['labels'][:31]
    result = {}
    for name, bank, rows in jobs:
        if mode == 'smoke':
            rows = rows[:17] if name == 'P10-3' else rows[:35]
            if name.endswith('/2'):
                continue
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        record = run(model, bank, rows, arm)
        torch.cuda.synchronize()
        record.update(seconds=perf_counter()-start,
                      allocated=torch.cuda.max_memory_allocated(),
                      reserved=torch.cuda.max_memory_reserved(), row_hash=rows_hash(rows))
        if name == 'P10-3' and arm == 'cap':
            assert record['labels'] == prior[name]['labels'][:len(rows)]
            if mode == 'full':
                assert record['activity'] == prior[name]['activity']
        result[name] = record
    assert 'neural-interfaces-2026/.venv/' not in Path('/proc/self/maps').read_text()
    Path(output).write_text(json.dumps(dict(
        gpu=torch.cuda.get_device_name(), arm=arm, mode=mode, streams=result)))


if __name__ == '__main__':
    main(*sys.argv[1:])
