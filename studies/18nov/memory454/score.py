"""Compare committed blind decisions with the frozen history443 stock reference."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from corrections import load
from fusion_inputs import official, probability, restore, sha
import score379 as ref

PREVIOUS = ref.WORK / 'ni26_history443_20261118'
SPEC = importlib.util.spec_from_file_location('history443_score', PREVIOUS / 'score.py')
previous = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(previous)
FROZEN = '46465f2c0ef5c5f984cd987cb0696e89ea0921a60a264d7d7b8ec131f0bb33e7'


def evaluate(kind, seed, smoke=False):
    assert ref.source_manifest(PREVIOUS/'source.sha256') == json.loads(
        (PREVIOUS/'379_before_labels.json').read_text())['source']
    assert sha(PREVIOUS/'379_decisions.npz') == FROZEN
    sc, archive, meta, ages = restore()
    frozen, order_age = ref.controls_and_age(meta)
    assert tuple(sc.ORDERS) == ref.ORDERS
    decisions, actor_hashes = {}, {}
    source_hash = sha(Path('actor_source.sha256'))
    active = ref.PEOPLE[:1] if smoke else ref.PEOPLE
    with np.load(PREVIOUS/'379_decisions.npz') as original:
        for p in active:
            idx = np.flatnonzero(meta['person'] == p)
            bank, actor_hashes[p] = load(Path.cwd(), kind, seed, p, idx)
            for oi, order in enumerate(ref.ORDERS):
                assert np.array_equal(order_age[oi], ages[order])
                prob = probability(archive['control',order,'B2_pool'][idx], ages[order][idx])
                with np.errstate(divide='ignore'):
                    base = np.log(prob)
                stock = base.argmax(1)
                assert np.array_equal(stock, original[f'{p}|control|{order}|stock'])
                assert np.array_equal(stock, frozen['frozen'][oi,idx])
                for arm in bank:
                    correction = bank[arm][oi]
                    chosen = (base + correction).argmax(1)
                    if arm == 'source':
                        assert np.all(correction == 0) and np.array_equal(chosen, stock)
                    decisions[f'{p}|{order}|{arm}'] = chosen.astype(np.uint8)
    if smoke:
        print('P01 all orders exact frozen + zero OK',flush=True)
        return
    output = Path(f'{kind}_{seed}_decisions.npz')
    np.savez_compressed(output, **decisions)
    proof = dict(decisions=sha(output), frozen443=FROZEN, actor_source=source_hash,
                 scorer_source=sha(Path('source.sha256')), actors=actor_hashes)
    output.with_suffix('.json').write_text(json.dumps(proof, indent=2))
    with np.load(ref.BASE/'data/proteus_windows.npz') as z:
        labels = z['y']
    metric = official()
    values = {a:np.empty((10,8)) for a in bank}
    changes = {a:0 for a in bank}
    counts = []
    for pi,p in enumerate(ref.PEOPLE):
        mask = meta['person'] == p
        counts.append(len(set(zip(meta['stream'][mask],meta['interface'][mask]))))
        for oi,order in enumerate(ref.ORDERS):
            for arm in bank:
                pred = decisions[f'{p}|{order}|{arm}']
                result = metric(labels[mask], pred, meta, mask)
                assert result['n_cells'] == counts[-1]
                values[arm][pi,oi] = 100 * result['balanced_accuracy_cell_mean']
                changes[arm] += int((pred != decisions[f'{p}|{order}|source']).sum())
    counts = np.array(counts)
    assert counts.sum() == 28
    scores = {a:previous.interval(table, counts)['pp'] for a,table in values.items()}
    assert abs(scores['source'] - 76.2793) < .0001
    contrasts = {f'memory-{a}':previous.interval(values['memory']-values[a],counts) for a in ('source','donor')}
    output.with_name(f'{kind}_{seed}_result.json').write_text(json.dumps(
        dict(kind=kind,seed=seed,scores=scores,changed=changes,contrasts=contrasts,**proof), indent=2))


if __name__ == '__main__':
    evaluate(sys.argv[1], int(sys.argv[2]), sys.argv[3:] == ['smoke'])
