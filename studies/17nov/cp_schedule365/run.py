"""Paired full-system replay of frozen, skipped and reassigned CP refits."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

from schedule import ARMS, apply, factories

OWNER = Path('/mnt/lustre/work/gehler/gfh527/ni26_soft_csp361_v1_20261117')
PINS = {'source.sha256': '9e889431c92f4759c79e0724446d19fc9503ac5b726464f73f9ed22b548f9ccd',
        'assets.py': 'd229c01316357b406ee7241a38aa1a6b371b0b5799f8eff005cbb786d298dd20',
        'measure.py': 'ad43e764a324f623337634a24ca50551e32afff8e3e9f1f274cc7df9869afec8'}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)


def owner():
    for name, expected in PINS.items():
        if digest(OWNER/name) != expected:
            raise ValueError(f'execution owner pin changed: {name}')
    sys.path.insert(0, str(OWNER))
    import assets
    import measure
    if (Path(assets.__file__) != OWNER/'assets.py'
            or Path(measure.__file__) != OWNER/'measure.py'):
        raise ValueError('execution owner proxy import')
    assets.verify_sources()
    return assets, measure.official()


def order_inputs(order, package, original):
    sessions = [package.C.order_rows(order, s, package.PROTO) for s in package.SESS]
    if len(sessions) != 14 or not np.array_equal(np.sort(np.concatenate(sessions)), np.arange(2655)):
        raise ValueError('not 14 complete pinned sessions')
    latch = package.latch(order)
    cache = package.J.hyp_cache(order, 'wprior', package.X, package.LAM,
                                sessions=package.SESS, proto=package.PROTO)
    with np.load(original/f'runs/chainout/s281_v66p_control_{order}.npz', allow_pickle=False) as z:
        raw = z['B2_pool|V6Fr'].astype(np.float64)
    filtered = package.gate(raw, package.J.filt_joint(raw, cache, 'urn'), order, latch)
    return raw, filtered, sessions


def pair(models, config, features, raw, filtered, sessions, expected, sequence):
    outputs, traces = {}, {}
    for arm in sequence:
        arr, trace = apply(models[arm], config, features, raw, filtered, sessions)
        if not np.isfinite(arr).all():
            raise ValueError(f'{arm} full replay outputs incomplete')
        outputs[arm], traces[arm] = arr, trace
    if not np.array_equal(outputs['frozen'], expected) or not np.array_equal(outputs['skip'], expected):
        raise ValueError('archive or omitted-refit exact guard failed')
    y = {k: v.argmax(1) for k, v in outputs.items()}
    changed = {}
    for arm in ('row_budget', 'recycle'):
        changed[arm] = int(np.count_nonzero(y[arm] != y['frozen']))
        if np.max(np.abs(outputs[arm]-outputs['frozen'])) <= 1e-6:
            raise ValueError(f'{arm} schedule had no numerical effect')
    for states in zip(*(traces[a]['sessions'] for a in ARMS), strict=True):
        if tuple(len(s['refits']) for s in states) != (9, 6, 7, 9):
            raise ValueError('complete session refit budget changed')
        if tuple(s['prefix_work'] for s in states) != (900, 780, 890, 1110):
            raise ValueError('registered prefix budget changed')
        if len({s['fusions'] for s in states}) != 1:
            raise ValueError('fusion exposure changed')
    return outputs, dict(traces=traces, decisions_changed=changed, sequence=sequence)


def state_test(parent, models, cfg, windows, features, raw, filt, sessions, outputs):
    s = sessions[0]
    native = parent(cfg)
    direct = np.stack([native.step(windows[row], raw[row], filt[row], False) for row in s])
    if not np.array_equal(direct, outputs['frozen'][s]) or native.log['err'] or native.dead:
        raise ValueError('immutable feature cache changes original CP')
    native.t = 0
    prefix = s[:150]
    for arm, factory in models.items():
        a, _ = apply(factory, cfg, features, raw, filt, (prefix,))
        b, _ = apply(factory, cfg, features, raw, filt, tuple(reversed(sessions[:2])))
        if not np.array_equal(a[prefix], outputs[arm][prefix]):
            raise ValueError('future windows changed the earlier decisions')
        idx = np.r_[sessions[0], sessions[1]]
        if not np.array_equal(b[idx], outputs[arm][idx]):
            raise ValueError('session reset/order isolation failed')
        c, trace = apply(factory, cfg, features, raw, filt, (s,), recognized=63)
        stats = trace['sessions'][0]
        if not np.array_equal(c[s], filt[s]) or stats['stopped'] != 63 or stats['fusions'] != 0:
            raise ValueError('early recognition output changed')


def score(metric, sc, data, outputs, metadata, person):
    with np.load(data, allow_pickle=False) as z:
        gold = z['y']
    ncell = np.empty(len(sc.PEOPLE), dtype=int)
    values = {a: np.empty((len(sc.ORDERS), len(sc.PEOPLE))) for a in ARMS}
    for oi, order in enumerate(sc.ORDERS):
        for pi, p in enumerate(sc.PEOPLE):
            mask = person == p
            for a in ARMS:
                estimate = metric(gold[mask], outputs[order][a][mask].argmax(1), metadata, mask)
                ncell[pi] = estimate['n_cells']
                values[a][oi, pi] = 100 * estimate['balanced_accuracy_cell_mean']
    values.update({f'{a}_minus_frozen': values[a]-values['frozen'] for a in ARMS[1:]})
    summaries = {
        a: dict(official=sc.cell_weighted_summary(v, sc.ORDERS, ncell),
                people_order=sc.summarize(v, sc.ORDERS, sc.PEOPLE))
        for a, v in values.items()
    }
    if abs(summaries['frozen']['official']['mean']-76.27933502145758) > 1e-7:
        raise ValueError('official control score changed')
    return summaries


def timing(traces):
    total = {a: Counter() for a in ARMS}
    for order in traces.values():
        for a, v in order['traces'].items():
            total[a]['cached_CP_seconds'] += v['seconds']
            for s in v['sessions']:
                total[a].update(dict(refit_seconds=s['refit_seconds'], refits=len(s['refits']),
                                     prefix_work=s['prefix_work'], selected_work=s['selected_work'],
                                     fusions=s['fusions']))
    return total


def run(smoke):
    start = time.monotonic()
    lock = digest('source.sha256')
    if not smoke:
        prior = json.loads(Path('smoke.json').read_text())
        if (prior['source_sha256'] != lock or not prior['passed']
                or prior['projected_full_seconds'] > 660 or prior['maxrss_kib'] > 3*1024**2):
            raise ValueError('smoke activity/resource envelope failed')
        if 'OK' not in Path('replay.log').read_text().splitlines():
            raise ValueError('missing required project replay')
    assets, metric = owner()
    parent, config = assets.get_stage()
    model = factories(parent, config)
    scorer, controls, person, _, metadata, windows = assets.load_controls()
    package = assets.get_count_filter()
    if not np.array_equal(package.ROWS, np.arange(len(windows))):
        raise ValueError('frozen count filter row ordering changed')
    returning = assets.returning_guard(scorer.ORDERS, int(config['v6_cp_w0']))
    stage = parent(config)
    features = np.stack([stage.feats(row) for row in windows])
    if features.shape != (2655, 8, 41, 41) or not np.isfinite(features).all():
        raise ValueError('frozen CP covariance schema differs or contains nonfinite rows')
    orders = ('within0',) if smoke else scorer.ORDERS
    traces, result = {}, {}
    timings = []
    for oi, order in enumerate(orders):
        tick = time.monotonic()
        raw, filt, sessions = order_inputs(order, package, assets.ORIGINAL)
        seq = ARMS[oi % len(ARMS):]+ARMS[:oi % len(ARMS)]
        expected = controls['control', order, 'B2_pool']
        out, trace = pair(model, config, features, raw, filt, sessions, expected, seq)
        timings.append(time.monotonic()-tick)
        result[order], traces[order] = out, trace
        print(f'{order}: frozen and skip bitwise; exact 9/6/7/9 refits per session', flush=True)
        if smoke:
            reverse, rev_trace = pair(model, config, features, raw, filt, sessions, expected, seq[::-1])
            if any(not np.array_equal(out[a], reverse[a]) for a in ARMS):
                raise ValueError('paired smoke warm replay differs')
            state_test(parent, model, config, windows, features, raw, filt, sessions, out)
            traces['reverse'] = rev_trace

    before = dict(passed=True, source_sha256=lock, elapsed_seconds=time.monotonic()-start,
                  maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  projected_full_seconds=12*max(timings)+45,
                  upstream='frozen v6.6cap250 B2_pool through official count filter',
                  returning=dict(archive_stop_positions=returning,
                                 last_stop=max(i for values in returning.values() for i in values),
                                 first_fusion=90, conditional_delta_pp=0),
                  traces=traces, work=timing(traces), held_outcomes_read=False)
    total = sum(v['decisions_changed']['recycle'] for v in traces.values())
    if not total:
        raise ValueError('recycled schedule never changed a decision')
    if smoke:
        write('smoke.json', before)
        print(json.dumps({k: before[k] for k in ('elapsed_seconds', 'work', 'projected_full_seconds',
                                                'maxrss_kib')}), flush=True)
        return
    if len(traces) != 8:
        raise ValueError('whole-chain matrix incomplete')
    assets.verify_sources()
    np.savez_compressed('scores.npz', **{f'{o}|{a}': v for o, arms in result.items()
                                        for a, v in arms.items()})
    before['scores_sha256'] = digest('scores.npz')
    write('before_labels.json', before)
    scores = score(metric, scorer, assets.CHAIN/'data/proteus_windows.npz', result, metadata, person)
    write('result.json', dict(metrics=scores, work=before['work'], source_sha256=lock,
                              changes={o: v['decisions_changed'] for o, v in traces.items()},
                              returning=before['returning'], scope='B2_pool new-person full-system DEV',
                              before_labels_sha256=digest('before_labels.json')))
    print(json.dumps(scores['recycle_minus_frozen']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--smoke', action='store_true')
    run(parser.parse_args().smoke)
