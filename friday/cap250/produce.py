"""Rebuild pinned cap250 count, routing and personal readout, then lock predictions."""
import json
import os
from pathlib import Path
import sys

import numpy as np
import score379 as ref
from assets import ORIGINAL, get_count_filter, get_stage, verify_sources

from count import emissions, terminal_scores

MATCH = ref.WORK / 'ni26_match456_stages_20261118'
MATCH_MANIFEST = 'a66ce105e52714a85d5214307799e13dcd5b989a45f7acd96024c3a2d1f99f1b'
RETURN = ref.WORK / 'ni26_return455_v3_20261118'
RETURN_REPORT = '77b4579e92a5f26036da14253c4e328c9df5b71f2930bd8e656bec5250a49b9a'
CANDIDATES = ('primary', 'history', 'confidence')
RETURN_INPUT = '5b6573ee9abe13a281916b6419b791215975dc2dc9122938d71f9dabcc269c9e'


def public_inputs():
    data = ref.BASE / 'data/proteus_windows.npz'
    perm = ref.BASE / 'runs/shuffle_perms.npz'
    assert ref.sha(data) == ref.DATA_SHA and ref.sha(perm) == ref.ORDER_SHA
    with np.load(data, allow_pickle=False) as d:
        meta = {k: d[k] for k in ('subject', 'interface', 'session')}
    meta['person'] = np.array([s.rsplit('/', 1)[-1] for s in meta['subject']])
    meta['stream'] = np.char.add(np.char.add(meta['subject'], '|'), meta['session'].astype(str))
    return meta, perm


def new_stock(order):
    assert ref.source_manifest(MATCH / 'source.sha256') == MATCH_MANIFEST
    path = MATCH / f'stages_{order}.npz'
    proof = json.loads(path.with_suffix('.json').read_text())
    assert proof['experiment'] == MATCH_MANIFEST and proof['order'] == order
    assert proof['output'] == ref.sha(path)
    archive = MATCH / 'stock' / f'{order}.npz'
    assert ref.sha(archive) == proof['archived_stock']
    with np.load(path) as z:
        raw, filtered, terminal = (z[k] for k in ('raw', 'filtered', 'terminal'))
    with np.load(archive) as z:
        assert np.array_equal(z['scores'], terminal)
    return raw, filtered, terminal, proof


def returning_stock(order, pkg, cache, latch, meta):
    receipt_file = RETURN / 'aux431_result.json'
    report_file = RETURN / 'result.json'
    assert ref.sha(receipt_file) == RETURN_INPUT and ref.sha(report_file) == RETURN_REPORT
    receipt = json.loads(receipt_file.read_text())
    report = json.loads(report_file.read_text())
    arm = RETURN / 'inputs' / f'{order}.npz'
    assert ref.sha(arm) == receipt['output_sha256'][order]
    with np.load(arm) as z:
        rows, raw, route = z['rows'], z['three'], json.loads(str(z['route']))
    assert np.array_equal(rows, pkg.ROWS) and len(rows) == 762
    assert route == receipt['recognition'][order]
    key = RETURN / f'{order}.npz'
    record = report['records'][order]
    assert ref.sha(key) == record['output']
    assert set(route) == set(meta['stream'][rows])
    for stream, stop in route.items():
        assert record['sessions'][stream]['stop'] == stop < 90
        assert record['sessions'][stream]['active'] == 0
    raw_bank = np.full((len(meta['stream']), 3), 1/3)
    raw_bank[rows] = raw
    filtered = pkg.gate(raw_bank, pkg.J.filt_joint(raw_bank, cache, 'urn'), order, latch)
    with np.load(key) as z:
        assert np.array_equal(z['rows'], rows)
        assert np.array_equal(z['stock'], filtered[rows].argmax(1))
    return raw_bank, filtered, filtered, dict(return_input=ref.sha(arm), stock=record['output'])


def produce(order, role, smoke):
    meta, permutation = public_inputs()
    verify_sources()
    _, config = get_stage()
    if role == 'new':
        pkg = get_count_filter()
        protocol = 'all_loso'
    else:
        assert role == 'returning' and 'PROTO' not in os.environ and 'SRC' not in os.environ
        os.environ['PROTO'] = protocol = 'seen_session'
        sys.path.insert(0, str(ORIGINAL / 'studies/14nov/rundetect'))
        import package as pkg
        assert Path(pkg.__file__).resolve() == ORIGINAL / 'studies/14nov/rundetect/package.py'
    assert pkg.PROTO == protocol and pkg.J.TAU == 2. and pkg.J.FLOOR == .5
    cache = pkg.J.hyp_cache(order, 'wprior', pkg.X, pkg.LAM, sessions=pkg.SESS, proto=protocol)
    latch = pkg.latch(order)
    if role == 'new':
        raw, filtered, terminal, source = new_stock(order)
    else:
        raw, filtered, terminal, source = returning_stock(order, pkg, cache, latch, meta)
    with np.load(permutation) as z:
        delivery = z[f'{order[:-1]}|{order[-1]}|{protocol}']
    assert np.array_equal(np.sort(delivery), np.sort(pkg.ROWS))

    stock_labels = terminal.argmax(1)
    assert np.array_equal(stock_labels, (terminal + 0.).argmax(1))
    decisions = {'stock': stock_labels, **{a: stock_labels.copy() for a in CANDIDATES}}
    eligible = np.zeros(len(raw), bool)
    for session, (rows, lp, _, reset, cap) in cache.items():
        stream = meta['stream'][rows[0]]
        assert np.array_equal(rows, delivery[meta['stream'][delivery] == stream])
        age = np.arange(len(rows))
        p, f, t = raw[rows], filtered[rows], terminal[rows]
        assert np.isfinite(p).all() and np.isfinite(f).all() and np.isfinite(t).all()
        assert np.allclose(p.sum(1), 1.) and np.allclose(f.sum(1), 1.)
        stop = latch[session] if latch[session] is not None else len(rows)
        assert np.array_equal(p[:stop], f[:stop])
        personal = age >= int(config['v6_cp_w0']) if role == 'new' else np.zeros(len(rows), bool)
        weight = float(config['v6_cp_cap']) * np.clip(
            (age-int(config['v6_cp_w0'])+1) / int(config['v6_cp_R']), 0., 1.
        )
        zero = terminal_scores(t, f, f, personal, weight)
        assert np.array_equal(zero, t)
        if stop == len(rows):
            continue
        eligible[rows[stop:]] = True
        stock, candidates = emissions(pkg.J.urn, p, lp, reset, cap)
        np.testing.assert_allclose(stock[stop:], f[stop:], atol=1e-12, rtol=0)
        if smoke:
            end = min(len(rows), stop + 17)
            short_stock, short = emissions(pkg.J.urn, p[:end], lp, reset[:end], cap[:end])
            np.testing.assert_allclose(short_stock, stock[:end], atol=1e-12, rtol=0)
            for name in candidates:
                np.testing.assert_allclose(short[name], candidates[name][:end], atol=1e-12, rtol=0)
        for name, values in candidates.items():
            values[:stop] = f[:stop]
            scores = terminal_scores(t, f, values, personal, weight)
            assert np.array_equal(scores[:stop], t[:stop])
            decisions[name][rows] = scores.argmax(1)

    print(role, order, 'eligible', int(eligible.sum()), 'decisions changed',
          {a: int(np.count_nonzero(v != stock_labels)) for a, v in decisions.items()}, flush=True)
    assert set(decisions) == {'stock', 'primary', 'history', 'confidence'}
    for values in decisions.values():
        assert np.array_equal(values[~eligible], stock_labels[~eligible])
    if smoke:
        assert np.count_nonzero(decisions['primary'] != stock_labels) > 0 or role == 'returning'
        return
    rows = np.sort(pkg.ROWS)
    dest = Path(f'{role}_{order}.npz')
    np.savez_compressed(dest, rows=rows, eligible=eligible[rows], **{a: v[rows] for a, v in decisions.items()})
    dest.with_suffix('.json').write_text(json.dumps(dict(
        output=ref.sha(dest), source=ref.sha(Path('source.sha256')), role=role, order=order,
        eligible=int(eligible.sum()), parents=source,
    )))


if __name__ == '__main__':
    produce(sys.argv[1], sys.argv[2], sys.argv[3:] == ['smoke'])
