"""Pin held-person frozen upstream inputs, then score saved native outputs."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path('/mnt/lustre/work/gehler/gfh527')
CAMPAIGN = ROOT/'ni26_cp_schedule365_v1_20261117'
sys.path.insert(0, str(CAMPAIGN))
from run import digest, order_inputs, owner, write

ARMS = ('stock', 'skip', 'phase90')


def capture(assets):
    scorer, controls, _, _, _, _ = assets.load_controls(dtype=np.float32)
    package = assets.get_count_filter()
    payload = {'orders': np.asarray(scorer.ORDERS)}
    for order in scorer.ORDERS:
        raw, filtered, sessions = order_inputs(order, package, assets.ORIGINAL)
        vectors = (np.concatenate(sessions), np.cumsum([0] + list(map(len, sessions))),
                   raw, filtered, controls['control', order, 'B2_pool'])
        for name, rows in zip(('index', 'bounds', 'raw', 'filtered', 'expected'), vectors, strict=True):
            payload[f'{order}|{name}'] = rows
    np.savez('inputs.npz', **payload)


def score(assets, metric):
    proof = json.loads(Path('before_labels.json').read_text())
    if proof['predictions'] != digest('outputs.npz') or proof['input'] != digest('inputs.npz'):
        raise ValueError('native prediction proof changed')
    scorer, _, person, _, metadata, _ = assets.load_controls(dtype=np.float32)
    data = {arm: np.empty((len(scorer.ORDERS), len(scorer.PEOPLE))) for arm in ARMS}
    counts = np.zeros(len(scorer.PEOPLE), dtype=int)
    with np.load(assets.CHAIN/'data/proteus_windows.npz', allow_pickle=False) as z:
        labels = z['y']
    with np.load('outputs.npz', allow_pickle=False) as out:
        for o, order in enumerate(scorer.ORDERS):
            for i, p in enumerate(scorer.PEOPLE):
                mask = person == p
                for arm in ARMS:
                    y = out[f'{order}|{arm}'][mask].argmax(1)
                    row = metric(labels[mask], y, metadata, mask)
                    counts[i] = row['n_cells']
                    data[arm][o, i] = 100*row['balanced_accuracy_cell_mean']
    data['phase90_minus_stock'] = data['phase90'] - data['stock']
    report = {arm: {'official': scorer.cell_weighted_summary(v, scorer.ORDERS, counts),
                    'people_order': scorer.summarize(v, scorer.ORDERS, scorer.PEOPLE)}
              for arm, v in data.items()}
    if abs(report['stock']['official']['mean']-76.27933502145758)>1e-7:
        raise ValueError('verified stock held-person score changed')
    write('score.json', dict(metrics=report, proof=proof, source_manifest=digest('source.sha256')))
    print(report['phase90_minus_stock'], flush=True)


def main(mode):
    assets, metric = owner()
    if mode == 'capture':
        capture(assets)
    elif mode == 'score' and 'OK' in Path('replay.log').read_text().splitlines():
        score(assets, metric)
    else:
        raise ValueError('unknown mode or missing project replay guard')


if __name__ == '__main__':
    main(*sys.argv[1:])
