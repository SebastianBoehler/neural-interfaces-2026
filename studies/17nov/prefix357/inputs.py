"""Pin the original unadapted decisions and extract pointwise support features."""
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path('/mnt/lustre/work/gehler/gfh527')
DIRECT = BASE / 'ni26_direct346_v1_20261117'
EXTEND = BASE / 'ni26_direct346_extension_v1_20261117'
sys.path.insert(0, str(DIRECT))
from common346 import (BASE as PROJECT, ORDERS, bank, digest, extract, labels, permutations,
                       restored, sha, source, state_sha)

FIT_PEOPLE = ('P04', 'P05', 'P06', 'P07', 'P08', 'P10')


def require(ok, text):
    if not ok:
        raise ValueError(text)


def pin(path, expected):
    require(sha(path) == expected, f'changed input: {path}')


def preflight():
    pin(DIRECT/'source.sha256', 'cc433f9b3f36dc381debb5b7a803784a67c5b397958e852cfecdbd36dc429dde')
    for line in (DIRECT/'source.sha256').read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        pin(DIRECT/name, expected)
    pin(EXTEND/'before_labels.json', 'dee726155ba28ab63e71d90e10ae47797a9c3870fb56fd908674099461108a84')
    proof = json.loads((EXTEND/'before_labels.json').read_text())
    require(proof['all_four_arms_both_people_eight_orders'] and proof['held_labels_opened'] is False,
            'missing historical by-order source proof')
    pin(EXTEND/'predictions.npz', proof['predictions_sha256'])
    pin(PROJECT/'runs/shuffle_perms.npz', '6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f')
    return proof


def prepare():
    proof = preflight()
    data, fit, held, people, _ = source()
    require(tuple(sorted(set(people[fit]))) == FIT_PEOPLE and set(people[held]) == {'P09'}
            and digest(fit) == proof['source_rows_sha256'], 'A6 body split mismatch')
    model, card = restored(data, fit, held)
    require(card['checkpoint_sha256'] == proof['checkpoint_sha256'], 'A6 body weights differ')
    state = state_sha(model.net)
    perms = permutations(len(people))
    arrays, row_masks, trace = {}, {}, {}
    with np.load(EXTEND/'predictions.npz', allow_pickle=False) as original:
        for person in ('P02', 'P03'):
            rows = np.flatnonzero(people == person)
            row_masks[person] = rows
            require(np.array_equal(original[person+'_index'], rows)
                    and len(set(data['stream'][rows])) == 1 and np.all(data['y'][rows] == -1),
                    'source trial/pinned single-session guard')
            require(not np.intersect1d(rows, fit).size and not np.intersect1d(rows, held).size,
                    'body/candidate/evaluation split collision')
            _, probability, _ = bank(model, data, rows, perms['within0'])
            expected = original[f'{person}_A_within0']
            error = float(np.max(np.abs(probability - expected)))
            require(error <= 1e-6 and np.array_equal(probability.argmax(1), expected.argmax(1)),
                    'original causal A checkpoint no longer matches')
            model.set_k(None)
            items = np.concatenate([extract(model, np.asarray([row]))[0] for row in rows])
            require(np.array_equal(items[0], extract(model, rows[:1])[0][0]),
                    'B1 population support depends on preceding items')
            model.set_k(20.)
            streams = {}
            for order in ORDERS:
                positions = np.searchsorted(rows, perms[order][np.isin(perms[order], rows)])
                require(np.array_equal(np.sort(positions), np.arange(len(rows))), 'order row mismatch')
                fixed = original[f'{person}_A_{order}']
                require(fixed.shape == (len(rows), 3) and np.isfinite(fixed).all() and (fixed > 0).all()
                        and np.allclose(fixed.sum(1), 1., atol=1e-12, rtol=0), 'invalid original current bank')
                streams[order] = (fixed[positions], items[positions], positions)
            arrays[person] = streams
            trace[person] = dict(rows_sha=digest(rows), support_sha=digest(items),
                                 original_probability_max_abs=error, rows=len(rows))
    require(state == state_sha(model.net), 'frozen A6 changed')
    return arrays, labels(row_masks['P02'], len(people)), trace, proof['checkpoint_sha256']
