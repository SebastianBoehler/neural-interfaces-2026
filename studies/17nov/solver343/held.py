"""Frozen original causal held inputs: features only, never held labels."""
import json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from bindings import sha
from verify import parity

BASE = Path('/mnt/lustre/work/gehler/gfh527/ni26_s301_native_reference_20261116/runs/s301_reference')


def compare(held, candidate, reference):
    allref = BASE.parents[2]/'ni26_s301_native_reference_all_20261116/runs/s301_reference_all'
    allmanifest = json.loads((allref/'manifest.json').read_text())
    reference_path = allref/f'{held}.npz'
    if sha(reference_path) != allmanifest['files'][f'{held}.npz']:
        raise ValueError('source fit reference hash differs')
    with np.load(reference_path, allow_pickle=False) as z:
        source_mean, source_scale, fit_rows = z['mean'].copy(), z['scale'].copy(), z['fit_rows'].copy()
    fixture_manifest = json.loads((BASE/'manifest.json').read_text())
    dataset = BASE.parents[1]/'data/proteus_windows.npz'
    with np.load(dataset, allow_pickle=False) as z:
        subject = z['subject']
    result = {}
    for order in ('within0', 'full0'):
        path = BASE/f'{held}_{order}.npz'
        if sha(path) != fixture_manifest['files'][path.name]:
            raise ValueError('frozen held fixture hash differs')
        with np.load(path, allow_pickle=False) as z:
            if not np.array_equal(z['mean'], source_mean) or not np.array_equal(z['scale'], source_scale) or not np.array_equal(z['fit_rows'], fit_rows):
                raise ValueError('held fixture scaler/source rows differ from actual fit')
            features = np.ascontiguousarray((z['expected_features']-z['mean'])/z['scale'])
            rows = z['rows'].copy()
        if len(rows) != len(features) or len(np.unique(rows)) != len(rows) or np.isin(rows, fit_rows).any() or not np.all(subject[rows] == f'Dreyer2026Proteus/{held}'):
            raise ValueError('query person/count/order/source exclusion differs')
        predicted = softmax(features @ candidate[:, :-1].T+candidate[:, -1], axis=1)
        expected = softmax(features @ reference[:, :-1].T+reference[:, -1], axis=1)
        row = parity(predicted, expected)
        row.update(exact_source_scaler=True, query_person=held, pinned_order=order, source_reference_sha256=sha(reference_path), permutation_sha256=fixture_manifest['permutation_sha256'], windows=len(features), input_sha256=sha(path), rows_sha256=__import__('hashlib').sha256(rows.tobytes()).hexdigest())
        row['passed'] = row['max_probability_error'] <= 1e-3 and row['non_tied_changes'] == 0
        result[order] = row
    return result
