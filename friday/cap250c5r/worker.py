"""Validate deferred covariance on identical RTX inputs, then time separate workers."""

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from qualify import Trial, empty, fresh_model, instrument, permutation, replay, stream, trials


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def case(order):
    channels, known = trials(5, True)
    with np.load('/external.npz', allow_pickle=False) as source:
        eeg = torch.from_numpy(source['X'])
        people, visit = source['subject'], source['session']
        assert source['ch_names'].tolist() == channels
    names = np.char.add(np.char.add(people, '|'), visit)
    who = 'Scherer2015/G|1'
    delivery = permutation(names, None, 'full', order)
    rows = delivery[names == who]
    reference = json.loads(Path('/rtx_g.json').read_text())
    assert reference['arm'] == 'cap' and reference['person'] == 5
    anchor = reference['sessions'][f'{who}|{order}']
    assert rows.tolist() == anchor['rows'] and len(rows) == 120
    known['unrecognized'] = Trial(who, eeg, rows, anchor)
    return channels, known


def shared(template, trial):
    model = fresh_model(template)
    empty(model)
    eager = model.cp
    deferred = deepcopy(eager)
    e, d = instrument(eager, False), instrument(deferred, True)
    old_step = eager.step
    scores, labels = [], []
    seconds = [0., 0.]

    def timed(step, slot, args):
        start = perf_counter()
        value = step(*args)
        seconds[slot] += perf_counter() - start
        return value

    def compare(*args):
        if len(scores) % 2:
            side = timed(deferred.step, 1, args)
            stock = timed(old_step, 0, args)
        else:
            stock = timed(old_step, 0, args)
            side = timed(deferred.step, 1, args)
        assert (stock.shape, stock.dtype, stock.tobytes()) == (side.shape, side.dtype, side.tobytes())
        scores.append(sha256(stock.dtype.str.encode()+stock.tobytes()).hexdigest())
        return stock

    eager.step = compare
    for row in trial.rows:
        label = model.predict(trial.eeg[int(row):int(row)+1].to('cuda:0'))
        assert label.shape == (1,) and label.dtype == torch.int64 and label.device.type == 'cpu'
        labels.append(int(label[0]))
    torch.cuda.synchronize()
    attrs = ('t', 'stopped', 'refits', 'fused', 'nfail', 'dead', 'log')
    assert all(getattr(eager, key) == getattr(deferred, key) for key in attrs) and not eager.dead
    assert all(e[key] == d[key] for key in ('selected', 'consumes', 'queries'))
    assert set(d['cov']) == set(d['consumes']) | set(d['queries']) and len(d['cov']) == len(set(d['cov']))
    for refit, chosen in enumerate(d['selected']):
        assert all(0 <= row < 20 * (refit + 1) for row in chosen)
    prefix = stream(template, trial, False, 17)
    assert prefix['labels'] == labels[:17] and prefix['terminal'] == scores[:17]
    assert scores and labels == trial.anchor['labels'][:len(labels)] and not model.dead
    activity = replay.activity(model)
    assert activity['windows'] == len(labels) and activity['errors'] == 0
    if len(labels) == len(trial.anchor['labels']):
        assert activity == trial.anchor['activity']
    return dict(session=trial.name, activity=activity, rows=trial.rows.tolist(), scores=scores,
                covariance=[len(e['cov']), len(d['cov'])], stage_seconds=seconds)


def profile(template, trial, order):
    schedule = (False, True, True, False) if order == 0 else (True, False, False, True)
    matched = {False: [], True: []}
    for deferred in schedule:
        warm = stream(template, trial, False, 17)
        record = stream(template, trial, deferred)
        assert warm['terminal'] == record['terminal'][:17]
        matched[deferred].append(record)
    a, b = matched[False], matched[True]
    assert all(record[key] == a[0][key] for record in a[1:] + b for key in ('rows', 'labels', 'work'))
    assert a[0]['cov'] == a[1]['cov'] > b[0]['cov'] == b[1]['cov']
    aa_delta = float(np.max(np.abs(np.asarray(a[0]['values']) - a[1]['values'])))
    ab_delta = max(float(np.max(np.abs(np.asarray(x['values']) - y['values']))) for x in a for y in b)
    verified = aa_delta <= 2e-5 and ab_delta <= min(2e-5, max(2e-7, 2 * aa_delta))
    timing_fields = ('wall_s', 'restore_s', 'first_s', 'p95_s', 'max_s', 'cov')
    result = dict(trial=trial.name, order=order, schedule=['deferred' if x else 'stock' for x in schedule],
                  independent_verified=verified, aa_delta=aa_delta, ab_delta=ab_delta,
                  covariance=[a[0]['cov'], b[0]['cov']])
    if verified:
        result.update(ratio=float(np.mean([x['wall_s'] for x in b]) / np.mean([x['wall_s'] for x in a])),
                      stock=[{k: x[k] for k in timing_fields} for x in a],
                      deferred=[{k: x[k] for k in timing_fields} for x in b])
    return result


def write(path, result):
    with path.open('x') as stream_out:
        stream_out.write(json.dumps(result, indent=2) + '\n')
    path.chmod(0o444)
    print(path.name, digest(path), flush=True)


def main(mode, order):
    device = torch.cuda.get_device_name()
    assert 'RTX 2080 Ti' in device and torch.__version__ == '2.14.1+cu130'
    source = digest('source.sha256')
    if mode == 'profile':
        smoke = json.loads(Path('smoke.json').read_text())
        assert smoke['source'] == source and smoke['shared_pass'] and not smoke['task_labels_opened']
    channels, cases = case(order)
    template = replay.load(Path('/package'), channels)
    for trial in cases.values():
        stream(template, trial, False, 17)
    shared_result = {role: shared(template, trial) for role, trial in cases.items()}
    assert shared_result['recognized']['covariance'] == [10, 0]
    original, reduced = shared_result['unrecognized']['covariance']
    assert 0 < reduced <= .80 * original and shared_result['unrecognized']['activity']['ft_steps'] > 0
    print('PASS same-input stage parity', {k: x['covariance'] for k, x in shared_result.items()}, flush=True)
    result = dict(mode=mode, order=order, source=source, device=device, shared_pass=True,
                  task_labels_opened=False, shared=shared_result)
    if mode == 'profile':
        result['profile'] = {role: profile(template, trial, order) for role, trial in cases.items()}
    file = 'smoke.json' if mode == 'smoke' else f'profile_{order}.json'
    assert 'neural-interfaces-2026/.venv/' not in Path('/proc/self/maps').read_text()
    write(Path(file), result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['smoke', 'profile'])
    parser.add_argument('order', type=int, choices=[0, 1])
    args = parser.parse_args()
    main(args.mode, args.order)
