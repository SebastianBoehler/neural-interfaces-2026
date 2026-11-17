"""Isolated full-chain fractional-CSP replay and independent zero guard."""
import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np

from assets import (ORIGINAL, get_count_filter, get_stage, load_controls, returning_guard,
                    require, sha, verify_sources)
from measure import summarize
from stage import apply, fractional_type


def write(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2)


def state_test(parent, candidate, config, x, features, raw, filtered, sessions):
    rows = sessions[0][:145]
    reference, dst = raw[rows], filtered[rows]
    selected = features[rows]
    local = np.arange(len(rows))
    actual = parent(config)
    direct = np.stack([actual.step(x[row], raw[row], filtered[row], False) for row in rows])
    require(actual.log['err'] == 0 and actual.fused > 0, 'real frozen CP smoke inactive')
    cached, _ = apply(parent, config, selected, reference, dst, [local])
    require(np.array_equal(direct, cached), 'fixed per-window feature cache changed CP outputs')
    result, _ = apply(candidate, config, selected, reference, dst, [local])
    prefix = min(125, len(rows)-1)
    short, _ = apply(candidate, config, selected[:prefix], reference[:prefix], dst[:prefix], [local[:prefix]])
    require(np.array_equal(result[:prefix], short), 'future windows affected emitted prefix')
    recognized, stats = apply(candidate, config, selected, reference, dst, [local], recognized=63)
    require(np.array_equal(recognized, dst) and stats[0]['fused'] == 0 and stats[0]['stopped'] == 63,
            'prior-to-fusion recognized route changed')
    two = np.r_[rows, sessions[1][:125]]
    split = (local, np.arange(len(local), len(two)))
    across, _ = apply(candidate, config, features[two], raw[two], filtered[two], split)
    reverse, _ = apply(candidate, config, features[two], raw[two], filtered[two], split[::-1])
    require(np.array_equal(across, reverse) and np.array_equal(across[:len(local)], result),
            'session reset/order isolation failed')


def apply_cell(parent, candidate, conf, features, raw, filt, sessions, expected):
    start = time.monotonic()
    p, zs = apply(parent, conf, features, raw, filt, sessions)
    q, qs = apply(candidate, conf, features, raw, filt, sessions)
    limit = float(np.max(np.abs(p-expected)))
    require(limit <= 1e-8 and np.array_equal(p.argmax(1), expected.argmax(1)), f'archived zero changed: {limit}')
    for a, b in zip(zs, qs):
        require(all(a[k] == b[k] for k in ('refits', 'fused', 'stopped', 'single_class')),
                'fractional arm changed refit/fusion schedule')
    changes = int(np.sum(p.argmax(1) != q.argmax(1)))
    require(sum(x['fractional_calls'] for x in qs) > 0
            and np.max(np.abs(p-q)) > 1e-6, 'fractional CSP silently inactive')
    return p, q, dict(seconds=time.monotonic()-start, zero_max_abs=limit, decisions_changed=changes,
                      unchanged_schedule=zs, changed_schedule=qs)


def run(smoke):
    start = time.monotonic()
    if not smoke:
        proof = json.loads(Path('smoke/proof.json').read_text())
        require(proof['source_sha256'] == sha(Path('source.sha256')) and proof['passed']
                and proof['production_projection_s'] < 2100 and proof['maxrss_kib'] < 6*1024**2,
                'replay cost/smoke guard absent')
        require('OK' in Path('replay.log').read_text().splitlines(), 'required project replay did not pass')
    verify_sources()
    parent, config = get_stage()
    candidate = fractional_type(parent)
    scorer, archive, people, cells, metadata, x = load_controls()
    package = get_count_filter()
    require(np.array_equal(package.ROWS, np.arange(len(x))) and np.array_equal(package.C.PER, people),
            'owning filter rows differ from archived control')
    stopped = returning_guard(scorer.ORDERS, int(config['v6_cp_w0']))
    template = parent(config)
    features = np.stack([template.feats(row) for row in x])
    require(features.shape == (len(x), 8, 41, 41) and np.isfinite(features).all(), 'window covariance schema')
    prefix_time = time.monotonic()-start
    output, traces, hypotheses_s, filter_s = {}, {}, [], []
    orders = ('within0',) if smoke else scorer.ORDERS
    for order in orders:
        tick = time.monotonic()
        sessions = [package.C.order_rows(order, s, package.PROTO) for s in package.SESS]
        flat = np.concatenate(sessions)
        require(np.array_equal(np.sort(flat), np.arange(len(x))), 'session delivery must partition all rows')
        latches = package.latch(order)
        cache = package.J.hyp_cache(order, 'wprior', package.X, package.LAM, sessions=package.SESS, proto=package.PROTO)
        hypotheses_s.append(time.monotonic()-tick)
        with np.load(ORIGINAL/f'runs/chainout/s281_v66p_control_{order}.npz', allow_pickle=False) as z:
            for base in (('B2_pool',) if smoke else scorer.BASES):
                tick = time.monotonic()
                raw = z[base+'|V6Fr'].astype(np.float64)
                filt = package.gate(raw, package.J.filt_joint(raw, cache, 'urn'), order, latches)
                filter_s.append(time.monotonic()-tick)
                p, q, info = apply_cell(parent, candidate, config, features, raw, filt, sessions,
                                        archive['control', order, base])
                output['control', order, base] = p
                output['fractional', order, base] = q
                traces[f'{order}|{base}'] = info
                if smoke:
                    state_test(parent, candidate, config, x, features, raw, filt, sessions)
        print(f'{order}: observed fractional CP and verified original zero', flush=True)
    elapsed = time.monotonic()-start
    estimate = prefix_time + 1.5*(8*max(hypotheses_s)+48*(max(filter_s)+max(t['seconds'] for t in traces.values()))) + 90
    result = dict(passed=True, source_sha256=sha(Path('source.sha256')), elapsed_s=elapsed,
                  production_projection_s=estimate if smoke else None, maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  scoring_labels_opened=False, archived_control_cells=len(traces), traces=traces,
                  returning_conditional=dict(method='all 32 natural sessions stop CP before any fusion; source proof and smoke, no whole rerun',
                                             stop_positions=stopped, delta_pp={'P04':0.,'P10':0.}))
    if smoke:
        Path('smoke').mkdir(exist_ok=False)
        require(next(iter(traces.values()))['decisions_changed'] > 0, 'real smoke has no changed labels')
        write('smoke/proof.json', result)
        print(json.dumps({k: v for k,v in result.items() if k not in ('traces','returning_conditional')}), flush=True)
        return
    require(len(traces)==48 and sum(v['decisions_changed'] for v in traces.values()) > 0, 'whole chain incomplete or inactive')
    verify_sources()
    np.savez_compressed('predictions.npz', **{'|'.join(key): val for key,val in output.items()})
    result['predictions_sha256'] = sha(Path('predictions.npz'))
    write('before_labels.json', result)
    quality = summarize(scorer, output, metadata, people)
    quality.update(before_labels_sha256=sha(Path('before_labels.json')), returning_conditional=result['returning_conditional'],
                   full_system_dev='new people, fold-specific unrecognized routing emulation, frozen cap250')
    write('result.json', quality)
    print(json.dumps(quality['all_bases']['fractional_minus_control']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--smoke', action='store_true')
    run(p.parse_args().smoke)
