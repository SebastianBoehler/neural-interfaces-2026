"""Verify exact complete-stack endpoints and time paired CP-stage streams."""
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import score379 as ref
from assets import ORIGINAL, get_stage, verify_sources
from produce import RETURN, RETURN_REPORT, new_stock, public_inputs, returning_stock

from lazy import detach, instrument


def inputs(order, role):
    meta, permutation = public_inputs()
    verify_sources()
    if role == 'new':
        raw, filtered, terminal, source = new_stock(order)
        protocol = 'all_loso'
        route = {}
    else:
        assert role == 'returning' and 'PROTO' not in os.environ and 'SRC' not in os.environ
        os.environ['PROTO'] = protocol = 'seen_session'
        sys.path.insert(0, str(ORIGINAL/'studies/14nov/rundetect'))
        import package as pkg
        assert pkg.PROTO == protocol and pkg.J.TAU == 2. and pkg.J.FLOOR == .5
        cache = pkg.J.hyp_cache(order, 'wprior', pkg.X, pkg.LAM, sessions=pkg.SESS, proto=protocol)
        raw, filtered, terminal, source = returning_stock(order, pkg, cache, pkg.latch(order), meta)
        report = RETURN/'result.json'
        assert ref.sha(report) == RETURN_REPORT
        record = json.loads(report.read_text())['records'][order]
        route = {name: v['stop'] for name, v in record['sessions'].items()}
        assert len(route) == 4 and all(0 <= v < 90 for v in route.values())
    with np.load(permutation, allow_pickle=False) as z:
        delivery = z[f'{order[:-1]}|{order[-1]}|{protocol}']
    with np.load(ref.BASE/'data/proteus_windows.npz', allow_pickle=False) as z:
        eeg = z['X']
    assert eeg.shape == (2655,41,480) and eeg.dtype == np.float32 and np.isfinite(eeg).all()
    expected = np.arange(len(eeg)) if role == 'new' else np.flatnonzero(np.isin(meta['stream'], list(route)))
    assert np.array_equal(np.sort(delivery), expected)
    streams = {k: delivery[meta['stream'][delivery] == k] for k in sorted(set(meta['stream'][delivery]))}
    assert len(streams) == (14 if role == 'new' else 4)
    return eeg, raw, filtered, terminal, source, streams, route


def trajectory(stage_type, config, eeg, raw, filtered, rows, stop, lazy):
    stage = stage_type(config)
    stats = instrument(stage, lazy)
    out = np.empty((len(rows),3))
    start = perf_counter()
    for i, row in enumerate(rows):
        out[i] = stage.step(eeg[row], raw[row], filtered[row], stop is not None and i >= stop)
    elapsed = perf_counter()-start
    assert stage.stopped == stop and stage.t == len(rows)
    assert not stage.dead and stage.log['err'] == 0 and np.isfinite(out).all()
    assert len(stats['selected']) == stage.refits
    assert len(stats['cov']) == len(set(stats['cov']))
    if lazy:
        assert set(stats['cov']) == set(stats['consumes']) | set(stats['queries'])
    else:
        assert stats['cov'] == list(range(len(rows) if stop is None else stop))
    counts = dict(refits=stage.refits, fused=stage.fused, failed=stage.nfail, stop=stage.stopped)
    assert elapsed > 0.
    detach(stage)
    return out, elapsed, stats, counts


def paired(stage_type, config, eeg, raw, filtered, terminal, rows, stop, smoke):
    args = (stage_type, config, eeg, raw, filtered, rows, stop)
    times = {False: [], True: []}
    proof = {}
    for sequence in ((False, True), (True, False)):
        results = {}
        for lazy in sequence:
            out, elapsed, stats, counts = results[lazy] = trajectory(*args, lazy)
            assert np.array_equal(out, terminal[rows])
            times[lazy].append(elapsed)
        _, _, stock, original = results[False]
        _, _, candidate, changed = results[True]
        assert original == changed
        for key in ('selected','consumes','queries'):
            assert stock[key] == candidate[key]
        seen = dict(counts=original, stock_cov=len(stock['cov']), lazy_cov=len(candidate['cov']))
        assert not proof or proof == seen
        proof = seen
    if smoke and stop is None:
        prefix = rows[: min(len(rows),117)]
        for lazy in (False, True):
            part, _, _, _ = trajectory(stage_type, config, eeg, raw, filtered, prefix, None, lazy)
            assert np.array_equal(part, terminal[prefix])
    return dict(**proof, stock_seconds=times[False], lazy_seconds=times[True], rows=len(rows))


def produce(order, role, smoke):
    stage_type, config = get_stage()
    eeg, raw, filtered, terminal, source, streams, route = inputs(order, role)
    if smoke:
        if role == 'new':
            key = next(iter(streams))
            streams = {key: streams[key]}
        else:
            selected = [min(route, key=route.get), max(route, key=route.get)]
            streams = {name: streams[name] for name in selected}
    records = {}
    for name, rows in streams.items():
        records[name] = paired(stage_type, config, eeg, raw, filtered, terminal, rows, route.get(name), smoke)
    stock = sum(v['stock_cov'] for v in records.values())
    lazy = sum(v['lazy_cov'] for v in records.values())
    print(order, role, 'exact vectors', sum(v['rows'] for v in records.values()),
          'covariance stock/candidate', stock, lazy, flush=True)
    if smoke:
        assert 0 <= lazy < stock
        return
    Path(f'{role}_{order}.json').write_text(json.dumps(dict(
        order=order, role=role, source=ref.sha(Path('source.sha256')),
        parents=source, records=records, exact=True,
    )))


if __name__ == '__main__':
    produce(sys.argv[1], sys.argv[2], sys.argv[3:] == ['smoke'])
