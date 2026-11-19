"""Read all original cap250 stages and replay the actual CP trajectory."""
import json
from pathlib import Path
import sys

import numpy as np
import score379 as ref
from assets import get_stage, verify_sources
from produce import RETURN, RETURN_REPORT, new_stock, public_inputs

from covariance import shape_only, use_shape_fit

ARMS = ('stock', 'shape_fit', 'shape_all')


def trajectory(stage_type, cfg, eeg, raw, filtered, rows, check=False):
    stages = {a: stage_type(cfg) for a in ARMS}
    stock_feats = stages['stock'].feats
    use_shape_fit(stages['shape_fit'])
    raw_features = unit_features = None
    stages['stock'].feats = stages['shape_fit'].feats = lambda _: raw_features
    stages['shape_all'].feats = lambda _: unit_features

    selected = {a: [] for a in ARMS}
    for name, stage in stages.items():
        def select(y, c, original=stage.select, out=selected[name]):
            rows = original(y, c)
            out.append(rows.copy())
            return rows
        stage.select = select

    scores = {a: np.empty((len(rows), 3), np.float64) for a in ARMS}
    spatial_delta = 0.
    for i, row in enumerate(rows):
        x, p, f = eeg[row], raw[row], filtered[row]
        raw_features = stock_feats(x)
        unit_features = shape_only(raw_features)
        if check:
            np.testing.assert_allclose(np.diagonal(unit_features, axis1=-2, axis2=-1).mean(-1),
                                       1., atol=1e-12, rtol=0)
        for name, stage in stages.items():
            scores[name][i] = stage.step(x, p, f, False)
        stock, candidate = stages['stock'].model, stages['shape_fit'].model
        if check and stock is not None and (i-cfg['v6_cp_t0']) % cfg['v6_cp_k'] == 0:
            assert candidate is not None
            w0, w1 = stock[0][0], candidate[0][0]
            q0, q1 = np.linalg.qr(w0)[0], np.linalg.qr(w1)[0]
            spatial_delta = max(spatial_delta, float(np.linalg.norm(q1@q1.T-q0@q0.T)))
    stock = stages['stock']
    assert stock.t == len(rows) and stock.stopped is None
    for name, stage in stages.items():
        assert (stage.fused, stage.refits, stage.nfail) == (stock.fused, stock.refits, stock.nfail)
        assert not stage.dead and stage.log['err'] == 0
        assert len(selected[name]) == len(selected['stock'])
        for a, b in zip(selected[name], selected['stock']):
            assert np.array_equal(a, b)
        assert np.isfinite(scores[name]).all()
        assert np.array_equal(scores[name][:cfg['v6_cp_w0']], scores['stock'][:cfg['v6_cp_w0']])
    return scores, spatial_delta


def produce(order, smoke):
    verify_sources()
    stage_type, config = get_stage()
    meta, permutation = public_inputs()
    raw, filtered, terminal, source = new_stock(order)
    with np.load(ref.BASE/'data/proteus_windows.npz') as z:
        eeg = z['X']
    with np.load(permutation) as z:
        delivery = z[f'{order[:-1]}|{order[-1]}|all_loso']
    assert np.array_equal(np.sort(delivery), np.arange(len(eeg)))
    returning = RETURN/'result.json'
    assert ref.sha(returning) == RETURN_REPORT
    records = json.loads(returning.read_text())['records']
    assert all(len(a['sessions']) == 4 for a in records.values())
    assert all(s['active'] == 0 and s['stop'] < config['v6_cp_w0']
               for r in records.values() for s in r['sessions'].values())
    sessions = sorted(np.unique(meta['stream']))
    if smoke:
        sessions = sessions[:1]
    scores = {a: terminal.copy() for a in ARMS}
    largest_spatial_delta = 0.
    for stream in sessions:
        rows = delivery[meta['stream'][delivery] == stream]
        outputs, delta = trajectory(stage_type, config, eeg, raw, filtered, rows, smoke)
        largest_spatial_delta = max(largest_spatial_delta, delta)
        assert np.array_equal(outputs['stock'], terminal[rows]), stream
        if smoke:
            n = config['v6_cp_w0'] + 19
            repeated, _ = trajectory(stage_type, config, eeg, raw, filtered, rows[:n], smoke)
            for name in ARMS:
                assert np.array_equal(outputs[name][:n], repeated[name]), name
        for name, values in outputs.items():
            scores[name][rows] = values
    assert np.array_equal(scores['stock'], terminal)
    assert np.array_equal((scores['stock']+0.).argmax(1), terminal.argmax(1))
    changed = {a: int(np.count_nonzero(s.argmax(1) != terminal.argmax(1))) for a, s in scores.items()}
    print(order, changed, 'largest spatial delta', largest_spatial_delta, flush=True)
    if smoke:
        assert largest_spatial_delta > 1e-5 and np.any(scores['shape_fit'] != terminal)
        assert np.any(scores['shape_all'] != scores['shape_fit'])
        return
    dest = Path(order+'.npz')
    np.savez_compressed(dest, **{a: s.argmax(1) for a, s in scores.items()})
    dest.with_suffix('.json').write_text(json.dumps(dict(
        output=ref.sha(dest), source=ref.sha(Path('source.sha256')), order=order,
        changed=changed, parents=source, returning_bank=RETURN_REPORT,
    )))


if __name__ == '__main__':
    produce(sys.argv[1], sys.argv[2:] == ['smoke'])
