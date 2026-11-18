"""Reconstruct the frozen historical count filter and apply native CP refinements."""
import json
from pathlib import Path
import sys

import numpy as np
import score379 as ref
from assets import ORIGINAL, get_count_filter, get_stage
from fusion_inputs import restore
from policy import corrections


def process(order, smoke=False):
    scorer, scores, meta, ages = restore()
    assert tuple(scorer.ORDERS) == ref.ORDERS
    parent, config = get_stage()
    assert tuple(config['v6_cp_'+k] for k in ('t0','k','w0','R','cap','q')) == (20,20,90,120,.10,.5)
    pkg = get_count_filter()
    input_file = ORIGINAL/f'runs/chainout/s281_v66p_control_{order}.npz'
    manifest = ref.BASE/'runs/coral304_inputs/control-original.sha256'
    assert ref.sha(manifest) == '42782d6c2b7de4a8adddfcc5282eadbc0294dd84185a31a2706a90856eb35660'
    expected = dict((k, h) for h,k in (line.split() for line in manifest.read_text().splitlines()))
    assert ref.sha(input_file) == expected[str(input_file.relative_to(ORIGINAL))]
    with np.load(input_file) as z:
        raw = z['B2_pool|V6Fr'].astype(np.float64)
    histories = pkg.J.hyp_cache(order, 'wprior', pkg.X, pkg.LAM, sessions=pkg.SESS, proto=pkg.PROTO)
    filtered = pkg.gate(raw, pkg.J.filt_joint(raw, histories, 'urn'), order, pkg.latch(order))
    sessions = [pkg.C.order_rows(order, s, pkg.PROTO) for s in pkg.SESS]
    for rows in sessions:
        assert np.array_equal(ages[order][rows], np.arange(len(rows)))
        assert len(set(meta['stream'][rows])) == 1
    terminal = scores['control',order,'B2_pool']
    controls, _ = ref.controls_and_age(meta)
    assert np.array_equal(terminal.argmax(1), controls['frozen'][ref.ORDERS.index(order)])
    shifts, error = corrections(raw, filtered, terminal, sessions)
    if smoke:
        rows = sessions[0][:137]
        short, _ = corrections(raw[rows], filtered[rows], terminal[rows], [np.arange(len(rows))])
        assert all(np.array_equal(v[rows],short[k]) for k,v in shifts.items())
        cp = parent(config)
        y = raw[rows[:120]].argmax(1)
        sel = cp.select(y, raw[rows[:120]].max(1))
        count = np.bincount(y,minlength=3)
        assert np.array_equal(np.bincount(y[sel],minlength=3),np.minimum(count,np.maximum(3,(count+1)//2)))
    decisions = {'stock': terminal.argmax(1).astype(np.uint8)}
    assert np.array_equal((terminal+0*shifts['prior']).argmax(1), decisions['stock'])
    for k,v in shifts.items():
        assert np.any(v) and np.isfinite(v).all()
        decisions[k] = (terminal+v).argmax(1).astype(np.uint8)
    if not smoke:
        path = Path(order+'.npz')
        np.savez_compressed(path, **decisions)
        path.with_suffix('.json').write_text(json.dumps(dict(bank=ref.sha(path), source=ref.sha(Path('source.sha256')),
                                                             raw=ref.sha(input_file), q_sum_max_error=error)))


if __name__ == '__main__':
    process(sys.argv[1], len(sys.argv)>2)
