"""Reproduce matched historical operands and seal causal decisions without gold."""
import json
import os
from pathlib import Path
import sys

import numpy as np
import score379 as ref
from assets import ORIGINAL, verify_sources
from produce import new_stock, public_inputs, returning_stock

from arbitrate import log_scores, variants

RET = ref.WORK/'ni26_aux431_20261118'
LINEAGE = RET/'evidence/next_wave_16nov/lineage419_result.json'
LINEAGE_SHA = '5e9d33d031feb47deabf8477acdf49266969b2611e0278b4b3ea478330c09593'
ARMS = ('stock', 'old', 'fixed', 'causal')


def archived_old(order, returning, pins):
    if returning:
        name = f'runs/chainout/ret_{order}.npz'
        path = RET/name
        key = order+'|B2_pool|base'
    else:
        path = ref.BASE/'runs/chainout'/f's281_v66p_control_{order}.npz'
        name = 'runs/lineage419/'+path.name
        key = 'B2_pool|P0r'
    digest = ref.sha(path)
    assert digest == pins[name]
    with np.load(path, allow_pickle=False) as z:
        old = z[key]
    return old,digest


def returning_inputs(order, pkg, meta):
    cache = pkg.J.hyp_cache(order,'wprior',pkg.X,pkg.LAM,sessions=pkg.SESS,proto=pkg.PROTO)
    _, _, terminal, source = returning_stock(order,pkg,cache,pkg.latch(order),meta)
    return terminal, source, pkg.ROWS


def produce(role, smoke):
    assert role in ('returning','new')
    returning = role == 'returning'
    meta, permutations = public_inputs()
    verify_sources()
    assert ref.sha(LINEAGE) == LINEAGE_SHA
    pins = json.loads(LINEAGE.read_text())['sha256']
    pkg = None
    protocol = 'all_loso'
    if returning:
        assert 'PROTO' not in os.environ and 'SRC' not in os.environ
        os.environ['PROTO'] = protocol = 'seen_session'
        sys.path.insert(0,str(ORIGINAL/'studies/14nov/rundetect'))
        import package as pkg
        assert pkg.PROTO == protocol and pkg.J.TAU == 2. and pkg.J.FLOOR == .5

    probes = ('within0','full0','runs0') if returning else ('within0',)
    smoke_switches = smoke_differences = 0
    for order in probes if smoke else ref.ORDERS:
        old, digest = archived_old(order,returning,pins)
        if returning:
            terminal,source,rows = returning_inputs(order,pkg,meta)
        else:
            _,_,terminal,source = new_stock(order)
            rows = np.arange(len(meta['stream']))
        assert old.shape == (len(rows),3) and len(rows) == (762 if returning else 2655)
        inverse = np.full(len(meta['stream']),-1)
        inverse[rows] = np.arange(len(rows))
        with np.load(permutations, allow_pickle=False) as z:
            perm = z[f'{order[:-1]}|{order[-1]}|{protocol}']
        assert np.array_equal(np.sort(perm),np.sort(rows))
        decisions = {a: np.full(len(rows),-1,dtype=np.int8) for a in ARMS}
        for stream in np.unique(meta['stream'][rows]):
            delivery = perm[meta['stream'][perm] == stream]
            where = inverse[delivery]
            assert np.array_equal(rows[where],delivery)
            ages = np.arange(len(delivery))
            v5 = log_scores(old[where], ages>=0)
            cap = log_scores(terminal[delivery], (ages<90) | returning)
            variants_real = variants(v5,cap)
            if smoke:
                stop = min(117,len(cap))
                prefix = variants(v5[:stop],cap[:stop])
                repeated = variants(v5,cap)
                zero = variants(cap,cap)
                for name,v in variants_real.items():
                    assert np.array_equal(v,repeated[name])
                    assert np.array_equal(v[:stop],prefix[name])
                    assert np.array_equal(zero[name],variants_real['stock'])
            for name,v in variants_real.items():
                decisions[name][where] = v
        assert all(np.isin(v,(0,1,2)).all() for v in decisions.values())
        assert np.array_equal(decisions['old'],old.argmax(1))
        assert np.array_equal(decisions['stock'],terminal[rows].argmax(1))
        switched = int(np.count_nonzero(decisions['causal'] != decisions['stock']))
        differing = int(np.count_nonzero(decisions['causal'] != decisions['fixed']))
        print(role,order,'causal switches',switched,'causal vs fixed',differing,flush=True)
        if smoke:
            smoke_switches += switched
            smoke_differences += differing
        else:
            dest=Path(f'{role}_{order}.npz')
            np.savez_compressed(dest,rows=rows,**decisions)
            dest.with_suffix('.json').write_text(json.dumps(dict(role=role,order=order,
                output=ref.sha(dest),source=ref.sha(Path('source.sha256')),
                old_sha256=digest,stock=source),indent=2))
    if smoke:
        assert smoke_switches > 0 and smoke_differences > 0


if __name__ == '__main__':
    produce(sys.argv[1],sys.argv[2:] == ['smoke'])
