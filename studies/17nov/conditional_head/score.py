"""Score only the completed guarded seven-source-person pilot."""
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prepare import OLD, ORDERS, QUERY, sha
METRICS = OLD/'2026-competition/tracks/bci_decoding/benchmark_utils/metrics.py'
if sha(METRICS) != 'fcc739400806a74259f248618501376a4fe53f2b5a40ccdc20db6a9e2369787d':
    raise ValueError('changed official metric helper')
sys.path.insert(0, str(METRICS.parent))
from metrics import balanced_accuracy, group_scores


def main():
    root = Path('runs/s326'); prep = json.loads((root/'prepare.json').read_text())
    guard = json.loads((root/'pilot-guards.json').read_text())
    if guard['query_labels_read_for_accuracy'] is not False or guard['prepare_sha256'] != sha(root/'prepare.json'):
        raise ValueError('incomplete source-only guard')
    if sha(root/'pilot-predictions.npz') != guard['predictions_sha256'] or sha(OLD/'data/proteus_windows.npz') != prep['data_sha256']:
        raise ValueError('prediction/data provenance mismatch')
    if any(guard['changed_labels'][k] <= 0 for k in ('moderate','strong')): raise ValueError('inactive mechanism')
    with np.load(root/'pilot-predictions.npz', allow_pickle=False) as predictions:
        with np.load(OLD/'data/proteus_windows.npz', allow_pickle=False) as z:
            y = z['y']; subject = np.asarray([str(v).rsplit('/',1)[-1] for v in z['subject']])
            sessions = z['session']; interfaces = z['interface']
        result = {}
        for person in QUERY:
            rows = predictions[f'{person}|rows']
            if not np.array_equal(rows,np.flatnonzero(subject==person)): raise ValueError('query rows changed')
            groups = np.column_stack((sessions[rows],interfaces[rows]))
            result[person] = {}
            for name in ('zero','moderate','strong'):
                values = {}
                for order in ORDERS:
                    pred = predictions[f'{person}|{order}|{name}']
                    if pred.shape != rows.shape or not np.isin(pred,(0,1,2)).all(): raise ValueError('bad predictions')
                    values[order] = float(100*group_scores(balanced_accuracy,y[rows],pred,groups).mean())
                values['scheme_equal'] = float(np.mean([np.mean([values[x] for x in ORDERS[:4]]),
                                                        np.mean([values[x] for x in ORDERS[4:6]]),
                                                        np.mean([values[x] for x in ORDERS[6:]])]))
                result[person][name] = values
    report = {'queries':QUERY, 'held_p01_used':False, 'source_guard_sha256':sha(root/'pilot-guards.json'),
              'results':result, 'scope':'seven-person source-only pilot; not a chain or outer-fold result'}
    with (root/'pilot-score.json').open('x') as f: json.dump(report,f,indent=2,sort_keys=True)
    print(json.dumps(report,sort_keys=True),flush=True)


if __name__ == '__main__': main()
