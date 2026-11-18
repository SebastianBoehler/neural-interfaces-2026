"""Require explicit-normalization support for a waveform model's proposed change."""
import json
import sys
from pathlib import Path

import numpy as np

owner = Path('/mnt/lustre/work/gehler/gfh527/ni26_fusion383_v1_20261118')
sys.path.insert(0, str(owner))
from experiment import interval
from fusion_inputs import BRANCHES, CHAIN, COHORT, FITS, official, pin, restore, sha

def decide(stock, learned, explicit):
    changed = learned != stock
    confirmed = changed & (explicit == learned)
    rest = changed & ~confirmed
    assert np.array_equal(confirmed | rest, changed)
    return np.where(confirmed, learned, stock), np.where(rest, learned, stock)

def evaluate(seed, smoke):
    proof = json.loads((owner / f'{seed}_before_labels.json').read_text())
    archive = owner / f'{seed}.npz'
    pin(archive, proof['predictions'])
    sc, _, meta, _ = restore()
    arms = ('stock', 'all', 'confirmed', 'unconfirmed', 'blank_confirmed')
    outputs = {}
    for branch in BRANCHES:
        scores = {a: np.empty((4, 8)) for a in arms}
        committed = []
        with np.load(archive) as predictions:
            for i, person in enumerate(COHORT):
                mask = meta['person'] == person
                for j, order in enumerate(sc.ORDERS):
                    prefix = f'{person}|{branch}|{order}|B2_pool|'
                    stock, learned, explicit, blank = (
                        predictions[prefix + a] for a in ('zero', 'J_real_5', 'K_5', 'J_blank_5'))
                    accepted, rejected = decide(stock, learned, explicit)
                    blank_accepted, _ = decide(stock, blank, explicit)
                    committed.append((i, j, mask, (stock, learned, accepted, rejected, blank_accepted)))
        if smoke:
            continue
        metric = official()
        with np.load(CHAIN / 'data/proteus_windows.npz') as data:
            labels = data['y']
        for i, j, mask, values in committed:
            for arm, pred in zip(arms, values):
                cell = metric(labels[mask], pred, meta, mask)
                assert cell['n_cells'] == 2
                scores[arm][i, j] = 100 * cell['balanced_accuracy_cell_mean']
        baseline = scores['stock']
        historical = json.loads((owner / f'{seed}_result.json').read_text())['data'][branch+'|B2_pool']
        assert abs(interval(baseline)['pp'] - historical['zero']) < 1e-9
        whole = scores['all'] - baseline
        assert abs(interval(whole)['pp'] - historical['contrasts']['J_real_5-zero']['pp']) < 1e-9
        np.testing.assert_allclose(scores['confirmed'] + scores['unconfirmed'] - 2 * baseline, whole, atol=1e-9)
        outputs[branch] = {a+'-stock': interval(scores[a]-baseline) for a in arms[1:]}
        outputs[branch]['confirmed-all'] = interval(scores['confirmed'] - scores['all'])
        outputs[branch]['confirmed-blank'] = interval(scores['confirmed'] - scores['blank_confirmed'])
    return outputs

if __name__ == '__main__':
    pin(owner / 'source.sha256', '9da18921a7e9d27807c21a511a7c27798c0f7f04f433833ce17419ac462c3522')
    for row in (owner / 'source.sha256').read_text().splitlines():
        digest, name = row.split(maxsplit=1)
        pin(owner / name, digest)
    if sys.argv[1] == 'smoke':
        for seed in FITS:
            evaluate(seed, True)
        Path('smoke.json').write_text(json.dumps({'source':sha(Path('source.sha256'))}))
    else:
        assert json.loads(Path('smoke.json').read_text())['source'] == sha(Path('source.sha256'))
        seed = int(sys.argv[1])
        Path(f'{seed}.json').write_text(json.dumps(evaluate(seed, False), indent=2))
