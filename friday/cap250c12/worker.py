"""One real exact-worker qualification of the directional and scalar histories."""
import hashlib
import json
from pathlib import Path
import resource
from time import perf_counter

import numpy as np
import torch

import replay
from benchmark_utils.streaming import fresh_model
from data import permutation
from history import ALIGNED, CAP, digest, install


ARMS = ('stock', 'zero', 'directional', 'scalar')
ACTIVE = ARMS[2:]
VISITS = ('Scherer2015/G|1', 'Scherer2015/G|2')
PACKAGE = Path('/package')


def stream(template, eeg, rows, arm):
    torch.cuda.synchronize()
    start = perf_counter()
    model = fresh_model(template)
    adapter = install(model, arm)
    labels, aligned, terminal, raw = [], [], [], []
    captured = []
    original = model.cp.step
    def observe(*args, **kw):
        p = original(*args, **kw)
        captured.append(p.copy())
        return p
    model.cp.step = observe
    for i in rows.tolist():
        p = model.predict(eeg[i:i+1].to('cuda:0'))
        assert p.device.type == 'cpu' and p.dtype == torch.int64 and p.shape == (1,)
        h = model.nets.last_inputs[ALIGNED[0]]
        assert all(model.nets.last_inputs[j] is h for j in ALIGNED)
        assert np.isfinite(h).all() and h.shape == (41, 480) and h.dtype == np.float32
        assert len(captured) == 1 and int(captured[0].argmax()) == int(p[0])
        assert not model.dead and not model.cp.dead and not model.cf.dead
        labels.append(int(p[0]))
        terminal.append(captured.pop())
        aligned.append(h.copy())
        raw.append(model.trace['out'].copy())
    torch.cuda.synchronize()
    seconds = perf_counter()-start
    info = replay.activity(model)
    assert info['windows'] == len(rows) and info['errors'] == info['ft_over'] == 0 and info['recognized'] == -1
    assert info['ft_refits'] > 0 and info['ft_steps'] == info['ft_refits']*20
    assert all(np.isfinite(p).all() for p in (*raw, *terminal))
    receipt = adapter.receipt() if adapter else None
    assert receipt is None or receipt['windows'] == len(rows)
    print(arm, len(rows), round(seconds, 3), info, 'clipped', len(receipt['active_ages']) if receipt else None, flush=True)
    return dict(seconds=seconds, labels=labels, aligned=np.asarray(aligned), raw=np.asarray(raw),
                terminal=np.asarray(terminal), activity=info, history=receipt)


def difference(a, b):
    x, y = a['aligned'], b['aligned']
    return dict(aligned_rows=int(np.count_nonzero(np.any(x != y, axis=(1, 2)))),
                max_aligned=float(np.abs(x-y).max()),
                raw_probability=float(np.abs(a['raw']-b['raw']).max()),
                terminal_probability=float(np.abs(a['terminal']-b['terminal']).max()),
                decisions=int(np.count_nonzero(np.asarray(a['labels']) != b['labels'])))


def assert_archive(output, archived):
    assert output['labels'] == archived['labels'] and output['activity'] == archived['activity']


def session_inputs():
    with np.load('/external.npz', allow_pickle=False) as z:
        eeg = torch.from_numpy(z['X'])
        people, days, channels = (z[k] for k in ('subject', 'session', 'ch_names'))
    assert eeg.shape == (2130, 41, 480) and eeg.dtype == torch.float32 and len(channels) == 41
    assert sorted(set(people))[5] == 'Scherer2015/G'
    archive = json.loads(Path('/prior/cap_5.json').read_text())
    names = np.char.add(np.char.add(people, '|'), days)
    rank = permutation(names, None, 'full', 411)
    rows = {s: rank[names == s] for s in VISITS}
    for s, first in zip(VISITS, (1257, 1383)):
        assert rows[s].tolist() == archive[s+'|411']['rows'] and len(rows[s]) == 120 and rows[s][0] == first
    return eeg, channels.tolist(), rows[VISITS[0]], rows[VISITS[1]], archive[VISITS[0]+'|411']


def validate_activity(rounds):
    metrics = {}
    for results in rounds:
        stock, zero = (results[a] for a in ARMS[:2])
        assert np.array_equal(stock['aligned'], zero['aligned'])
        assert stock['labels'] == zero['labels'] and stock['activity'] == zero['activity']
        histories = [results[a]['history'] for a in ACTIVE]
        assert histories[0]['active_ages'] == histories[1]['active_ages']
        assert np.array_equal(histories[0]['retained_traces'], histories[1]['retained_traces'])
        assert histories[0]['prior'] == histories[1]['prior'] == zero['history']['prior']
        active = len([age for age in histories[0]['active_ages'] if age < 119])
        volume = histories[0]['directional_scalar_cov_max']
        assert volume > 0 or active == 0
        pair = {a: difference(results[a], stock) for a in (*ACTIVE, 'zero')}
        pair['directional_minus_scalar'] = difference(*(results[a] for a in ACTIVE))
        for a in ACTIVE:
            res = results[a]
            assert np.array_equal(stock['aligned'][0], res['aligned'][0]) and stock['labels'][0] == res['labels'][0]
            assert stock['activity']['ft_refits'] == res['activity']['ft_refits']
            assert stock['activity']['ft_steps'] == res['activity']['ft_steps']
            if histories[0]['active_ages']:
                first = histories[0]['active_ages'][0]
                assert np.array_equal(stock['aligned'][:first+1], res['aligned'][:first+1])
        gate = 2 <= active <= 60 and all(pair[a]['aligned_rows'] >= 6 and pair[a]['terminal_probability'] > 1e-5
                                         for a in ACTIVE) and pair['directional_minus_scalar']['aligned_rows'] > 0
        metrics[len(metrics)] = dict(past_active_updates=active, contrasts=pair, activation_passed=gate)
    return metrics


def summarize(r):
    return {**{k: v for k, v in r.items() if k not in ('aligned', 'raw', 'terminal')},
            **{k+'_sha256': digest(r[k]) for k in ('aligned', 'raw', 'terminal')}}


def main():
    device = torch.cuda.get_device_name()
    assert torch.__version__ == '2.14.1+cu130' and 'A100' in device
    eeg, ch, rows, other, archive = session_inputs()
    torch.cuda.synchronize()
    load_start = perf_counter()
    template = replay.load(PACKAGE, ch)
    torch.cuda.synchronize()
    load_seconds = perf_counter()-load_start
    pre = stream(template, eeg, rows, 'stock')
    assert_archive(pre, archive)
    orders = (ARMS, ARMS[::-1])
    rounds = [{a: stream(template, eeg, rows, a) for a in schedule} for schedule in orders]
    for r in rounds:
        for a in ARMS[:2]:
            assert_archive(r[a], archive)
    activation = validate_activity(rounds)
    for a in ARMS:
        first = rounds[0][a]
        repeated = rounds[1][a]
        assert first['labels'] == repeated['labels'] and np.array_equal(first['aligned'], repeated['aligned'])
        stream(template, eeg, other[:31], a)
        prefix = stream(template, eeg, rows[:31], a)
        assert first['labels'][:31] == prefix['labels'] and np.array_equal(first['aligned'][:31], prefix['aligned'])
    timings = {a: [r[a]['seconds']/r['stock']['seconds'] for r in rounds] for a in (*ACTIVE, 'zero')}
    timer_passed = all(np.median(timings[a]) <= 1.05 and max(timings[a]) <= 1.10 for a in ACTIVE)
    assert not template.v5hist and template.cp.t == 0
    assert 'neural-interfaces-2026/.venv/' not in Path('/proc/self/maps').read_text()
    summary = dict(device=device, cap=CAP, no_gold=True, load_seconds=load_seconds,
                   replay='live exact frozen cap250 G visit 1 order 411', rows=rows.tolist(),
                   controls_passed=True, reset_passed=True, activation=activation, warm_ratios=timings,
                   a100_budget_passed=bool(timer_passed), pending_rtx=True,
                   qualification_passed=bool(timer_passed and all(v['activation_passed'] for v in activation.values())),
                   rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   source_sha256=hashlib.sha256(Path('source.sha256').read_bytes()).hexdigest(),
                   rounds=[{a: summarize(r[a]) for a in ARMS} for r in rounds])
    path = Path('ea_smoke.json')
    with path.open('x') as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print('F-C12 controls passed', 'activation', activation, 'warm ratios', timings, 'advanced', summary['qualification_passed'], flush=True)


if __name__ == '__main__':
    main()
