"""Time the complete staged worker with the instrumented F-C5 personalizer."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import resource
import sys
from time import perf_counter

import numpy as np
import torch

from data import permutation, sha
from lazy import detach, instrument
import replay
from benchmark_utils.streaming import fresh_model

CASES = (('P01-1', 0, 'A'), ('P04-2', 2, 'D'), ('P10-3', 5, 'G'))
SEEDS = (411, 1411)


@dataclass
class Trial:
    name: str
    eeg: torch.Tensor
    rows: np.ndarray
    anchor: dict


def trials(index, smoke):
    pub, person, letter = CASES[index // 2]
    seed = SEEDS[index % 2]
    x, sessions, channels = replay.read_source()
    pub_anchor = json.loads(Path('/normal.json').read_text())['streams']['first'][pub]
    ordinary = Trial(pub, x, dict(sessions)[pub], pub_anchor)
    with np.load('/external.npz', allow_pickle=False) as z:
        y = torch.from_numpy(z['X'])
        owners, days = (z[k] for k in ('subject', 'session'))
        assert z['ch_names'].tolist() == channels
    assert y.shape == (2130, 41, 480) and y.dtype == torch.float32
    groups = np.char.add(np.char.add(owners, '|'), days)
    who = f'Scherer2015/{letter}'
    name = f'{who}|1|{seed}'
    assert sorted(set(owners))[person] == who
    external = json.loads(Path(f'/prior/cap_{person}.json').read_text())[name]
    p = permutation(groups, None, 'full', seed)
    selected = p[(owners == who)&(days == '1')]
    assert selected.tolist() == external['rows'] and len(selected) == 120
    unfamiliar = Trial(name, y, selected, external)
    if smoke:
        ordinary.rows = ordinary.rows[:31]
    return channels, dict(recognized=ordinary, unrecognized=unfamiliar)


def empty(model):
    stage = model.cp
    assert type(stage).__name__ == 'CausalPers'
    assert not stage.t and not stage.C and not stage.P and stage.model is None
    assert stage.stopped is None and not stage.dead and not stage.log['err']
    assert not model.v5hist and model.log6['n'] == 0 and not model.dead
    assert 'feats' not in vars(stage) and 'step' not in vars(stage)


def stream(template, trial, deferred, count=None):
    rows = trial.rows if count is None else trial.rows[:count]
    torch.cuda.synchronize()
    start = perf_counter()
    model = fresh_model(template)
    empty(model)
    stage = model.cp
    stats = instrument(stage, deferred)
    old_step = stage.step
    terminal = []
    def capture(*args):
        output = old_step(*args)
        terminal.append(hashlib.sha256(output.dtype.str.encode()+output.tobytes()).hexdigest())
        return output
    stage.step = capture
    ready = perf_counter()
    labels, times = [], []
    for row in rows:
        t = perf_counter()
        answer = model.predict(trial.eeg[int(row):int(row)+1].to('cuda:0'))
        assert answer.device.type == 'cpu' and answer.dtype == torch.int64 and answer.shape == (1,)
        labels.append(int(answer[0]))
        times.append(perf_counter()-t)
    torch.cuda.synchronize()
    finish = perf_counter()
    assert not any(p.dead for p in (model, stage, model.cf))
    activity = replay.activity(model)
    assert labels == trial.anchor['labels'][:len(rows)] and len(terminal) == len(rows)
    assert activity['windows'] == len(rows) and activity['errors'] == 0
    if len(rows) == len(trial.anchor['labels']):
        assert activity == trial.anchor['activity']
    assert stage.stopped == (None if activity['recognized'] == -1 else activity['recognized'])
    assert len(stats['selected']) == stage.refits
    assert len(stats['cov']) == len(set(stats['cov']))
    assert set(stats['cov']) == (set(stats['consumes'])|set(stats['queries']) if deferred
                                else set(range(stage.t if stage.stopped is None else stage.stopped)))
    work = dict(activity=activity, failed=stage.nfail, stop=stage.stopped,
                **{key: stats[key] for key in ('selected','consumes','queries')})
    del stage.step
    detach(stage)
    empty(template)
    return dict(wall_s=finish-start, restore_s=ready-start, first_s=times[0],
                p95_s=float(np.quantile(times,.95)), max_s=max(times),
                rows=rows.tolist(), labels=labels, terminal=terminal, work=work, cov=len(stats['cov']))


def main(mode, index, arm, out):
    assert torch.__version__ == '2.14.1+cu130' and 'A100' in torch.cuda.get_device_name()
    assert mode in ('smoke','produce') and arm in ('stock','deferred')
    deferred = arm == 'deferred'
    start = perf_counter()
    channels, series = trials(index, mode == 'smoke')
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    loaded = perf_counter()
    template = replay.load(Path('/package'), channels)
    torch.cuda.synchronize()
    load_s = perf_counter()-loaded
    roles = tuple(series) if index % 2 == 0 else tuple(reversed(series))
    warm = {}
    if mode == 'produce':
        for role in roles:
            warm[role] = stream(template, series[role], deferred, 17)
    records = {}
    for role in roles:
        if mode == 'smoke':
            warm[role] = stream(template, series[role], deferred, 1)
        row = records[role] = stream(template, series[role], deferred)
        assert row['work']['activity']['recognized'] == (10 if role == 'recognized' else -1)
        if mode == 'smoke':
            again = stream(template, series[role], deferred, 17)
            assert all(again[k] == row[k][:17] for k in ('rows','labels','terminal'))
        assert all(warm[role][k] == row[k][:len(warm[role]['labels'])] for k in ('rows','labels','terminal'))
        print(role, row['wall_s'], row['cov'], row['work']['activity'], flush=True)
    assert 'neural-interfaces-2026/.venv/' not in Path('/proc/self/maps').read_text()
    result = dict(mode=mode, index=index, arm=arm, source=sha(Path('source.sha256')),
                  gpu=torch.cuda.get_device_name(), gpu_live_bytes=torch.cuda.max_memory_allocated(),
                  gpu_reserved_bytes=torch.cuda.max_memory_reserved(), load_s=load_s,
                  process_s=perf_counter()-start, rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  records=records, warm=warm)
    path = Path(out)
    path.write_text(json.dumps(result)+'\n')
    path.with_suffix('.sha256').write_text(sha(path)+'\n')


if __name__ == '__main__':
    mode, index, arm, out = sys.argv[1:]
    main(mode, int(index), arm, out)
