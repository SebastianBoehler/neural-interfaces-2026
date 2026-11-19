"""Replay the actual staged Solver on canonical Scherer sessions."""
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
import replay
from benchmark_utils.streaming import fresh_model
from data import permutation

from weights import policy_weights


def instrument(model, arm):
    ft = model.ft
    assert model.routed == 1 and model.r15 and not model.v6r
    assert ft.on and ft.n_steps == 5 and ft.t0 == 20 and ft.every == 10
    assert len(ft.member) == 4 and ft.cap == 250
    original = ft._refit
    events = []

    def fit(t, labels, original_weights):
        assert not model.recB and ft.relabels == ft.refits+1
        first = np.asarray(ft.lab[:t])
        current = np.asarray(labels)
        weights = np.asarray(original_weights, dtype=np.float64)
        assert len(ft.lab) == t and first.shape == current.shape == weights.shape == (t,)
        assert np.isin(first, [0, 1, 2]).all() and np.isin(current, [0, 1, 2]).all()
        assert np.isfinite(weights).all() and (weights > 0).all()
        adjusted, disagreement = policy_weights(arm, first, current, weights)
        changed = np.count_nonzero(adjusted.astype(np.float32) != weights.astype(np.float32))
        events.append(dict(age=t, changed=int(changed), disagreement=disagreement))
        original(t, labels, original_weights if arm == 'stock' else adjusted)

    ft._refit = fit
    return events, original


def stream(template, eeg, rows, arm):
    torch.cuda.synchronize()
    started = perf_counter()
    model = fresh_model(template)
    events, original = instrument(model, arm)
    labels = []
    for row in rows:
        label = model.predict(eeg[int(row):int(row)+1].to('cuda:0'))
        assert label.device.type == 'cpu' and label.dtype == torch.int64 and label.shape == (1,)
        labels.append(int(label[0]))
    torch.cuda.synchronize()
    activity = replay.activity(model)
    assert activity['errors'] == 0 and not model.dead and activity['windows'] == len(rows)
    assert model.ft.dtheta > 0 and len(events) == activity['ft_refits'] == activity['ft_relabels']
    assert activity['ft_steps'] == 20*len(events)
    assert [v['age'] for v in events] == list(range(20, 20+10*len(events), 10))
    if activity['recognized'] >= 0:
        assert all(v['age'] < activity['recognized'] for v in events)
    model.ft._refit = original
    del original, model
    torch.cuda.synchronize()
    return dict(labels=labels, activity=activity, events=events, seconds=perf_counter()-started)


def run(index, arm, smoke):
    assert torch.__version__ == '2.14.1+cu130' and 'A100' in torch.cuda.get_device_name()
    with np.load('/external.npz', allow_pickle=False) as z:
        people, days, channels = (z[k] for k in ('subject', 'session', 'ch_names'))
        eeg = torch.from_numpy(z['X'])
    assert eeg.shape == (2130, 41, 480) and eeg.dtype == torch.float32
    who = sorted(set(people))[index]
    sessions = np.char.add(np.char.add(people, '|'), days)
    references = json.loads(Path(f'/prior/cap_{index}.json').read_text())
    template = replay.load(Path('/package'), channels.tolist())
    output = {}
    for seed in (0,) if smoke else (0, 1):
        order = permutation(sessions, None, 'full', seed)
        for day in sorted(set(days[people == who])):
            key = f'{who}|{day}|{seed}'
            reference = references[key]
            rows = order[(people == who) & (days == day)]
            assert rows.tolist() == reference['rows']
            if smoke:
                rows = rows[:61]
            result = stream(template, eeg, rows, arm)
            assert result['labels'][:20] == reference['labels'][:20]
            if arm == 'stock':
                assert result['labels'] == reference['labels'][:len(rows)], key
            if not smoke:
                assert result['activity'] == reference['activity'], (key, arm, result['activity'])
            output[key] = dict(**result, rows=rows.tolist())
    if not smoke:
        first = next(iter(output.values()))
        warm = stream(template, eeg, first['rows'], arm)
        assert warm['activity'] == first['activity']
        first['cold_seconds'], first['seconds'] = first['seconds'], warm['seconds']
        first['warm_differences'] = np.flatnonzero(np.array(first['labels']) != warm['labels']).tolist()
    count = sum(e['disagreement'] for v in output.values() for e in v['events'])
    changed = sum(e['changed'] for v in output.values() for e in v['events'])
    if smoke:
        assert count >= 5
        assert (changed > 0) == (arm != 'stock')
    assert 'neural-interfaces-2026/.venv/' not in Path('/proc/self/maps').read_text()
    Path(f'{arm}_{index}{"_smoke" if smoke else ""}.json').write_text(json.dumps(output))
    print(index, arm, len(output), 'unstable targets', count, 'weights changed', changed, flush=True)


if __name__ == '__main__':
    run(int(sys.argv[1]), sys.argv[2], sys.argv[3:] == ['smoke'])
