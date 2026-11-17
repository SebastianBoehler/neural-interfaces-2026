"""Produce every source-held and external prediction before external accuracy."""
import argparse
import json
import pickle
import resource
import time
from pathlib import Path

import numpy as np

from data import ORDERS, load_source, load_target, permutation, sha, source_orders
from geometry import Geometry, digest, fit, predict


def banks(bank, ids, stream, orders, prior, models):
    result = {f'{arm}|{name}': np.full((len(stream), 2), np.nan)
              for arm in models for name in orders}
    select = np.zeros(len(stream), bool)
    select[ids] = True
    before = digest(models)
    for oi, (name, order) in enumerate(orders.items()):
        sequence = order[select[order]]
        for session in dict.fromkeys(stream[sequence]):
            rows = sequence[stream[sequence] == session]
            values = predict(models, bank.causal(rows, prior))
            if oi == 0:
                short = predict(models, bank.causal(rows[:17], prior))
                assert all(np.array_equal(values[k][:17], short[k]) for k in models)
                # A separate stream interleaved between repeated calls cannot change predictions.
                other = sequence[stream[sequence] != session][:7]
                if len(other):
                    predict(models, bank.causal(other, prior))
                repeated = predict(models, bank.causal(rows, prior))
                assert all(np.array_equal(values[k], repeated[k]) for k in models)
            for arm, p in values.items():
                result[f'{arm}|{name}'][rows] = p
    assert digest(models) == before, 'Inference changed fitted model'
    assert all(np.isfinite(p[ids]).all() for p in result.values())
    return result


def run(root, out, smoke):
    t, cpu = time.monotonic(), time.process_time()
    out.mkdir(parents=True, exist_ok=False)
    smoke = json.loads(smoke.read_text())
    source = load_source(root)
    assert sha(root / 'data/proteus_windows.npz') == smoke['source_data_sha256']
    orders = source_orders(root, source)
    target = load_target(root)  # Never requests target label member.
    expected = json.loads((root / 'runs/diag/external_semantics351/audit.json').read_text())
    assert sha(root / 'data/scherer2015/windows.npz') == expected['scherer']['source_sha256']
    n = len(source['X'])
    x = np.concatenate([source['X'], target['X']])
    bank = Geometry(x, root)
    del x
    stream = np.r_[source['stream'], target['stream']]
    all_source = np.arange(n)
    source_banks = {f'{arm}|{order}': np.full((n, 2), np.nan)
                    for arm in ('power', 'cov') for order in orders}
    exclusions = []
    for person in sorted(set(source['subject'])):
        train = np.flatnonzero(source['subject'] != person)
        query = np.flatnonzero(source['subject'] == person)
        assert not set(train) & set(query)
        features = bank.training(train, stream)
        models = fit(features, source['y'][train])
        assert digest(models) == digest(fit(features, source['y'][train]))
        predicted = banks(bank, query, stream[:n], orders, bank.prior(train), models)
        for key in source_banks:
            source_banks[key][query] = predicted[key][query]
        exclusions.append({'person': person, 'training_people': sorted(set(source['subject'][train])),
                           'train_rows_sha256': __import__('hashlib').sha256(train.tobytes()).hexdigest(),
                           'model_sha256': digest(models)})
        print('source_fold_complete', person, flush=True)
    assert all(np.isfinite(p).all() for p in source_banks.values())
    np.savez_compressed(out / 'source_predictions.npz', **source_banks)
    models = fit(bank.training(all_source, stream), source['y'])
    assert digest(models) == digest(fit(bank.training(all_source, stream), source['y']))
    (out / 'frozen_source_models.pkl').write_bytes(pickle.dumps(models))
    target_ids = np.arange(n, len(stream))
    target_orders = {f'{s}{seed}': n + permutation(target['stream'], target['group'], s, seed)
                     for s, seed in ORDERS}
    target_predictions = banks(bank, target_ids, stream, target_orders, bank.prior(all_source), models)
    np.savez_compressed(out / 'target_predictions.npz',
                        **{key: p[target_ids] for key, p in target_predictions.items()})
    np.savez_compressed(out / 'target_orders.npz', **target_orders)
    result = {'source_exclusions': exclusions, 'target_labels_opened': False,
              'source_rows': n, 'target_rows_all_tasks': len(target_ids), 'orders': list(orders),
              'source_model_sha256': digest(models),
              'prefix_reset_interleaving_and_immutable_guards': True,
              'inputs': {str(p): sha(root / p) for p in ('data/proteus_windows.npz',
                         'data/scherer2015/windows.npz', 'runs/shuffle_perms.npz',
                         'runs/diag/external_semantics351/scherer_rows.tsv.gz')},
              'source_hashes': {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')},
              'outputs': {p.name: sha(p) for p in out.iterdir() if p.is_file()},
              'wall_s': time.monotonic() - t, 'cpu_s': time.process_time() - cpu,
              'maxrss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (out / 'before_labels.json').write_text(json.dumps(result, indent=2) + '\n')
    print('BEFORE_LABELS_ALL_BANKS_PASS', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--smoke', type=Path, required=True)
    a = p.parse_args()
    run(a.root, a.out, a.smoke)
