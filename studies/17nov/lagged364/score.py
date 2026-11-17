"""One official metric pass after complete chain, order, and zero checks."""
import importlib.util
import json
from pathlib import Path

import numpy as np

from chain import FAMILIES, HERE, OLD, ORDERS, ROOT, ROWS, load_scorer, sha, validate

METRIC = HERE.parent / 'ni26_raw312_cpu_20261117_v2/experiments/protocol.py'
METRIC_SHA = '390fd6bea0ddc9c53e34fdabb094b1a269ff45b07b4a5542a8135b44d5314a2c'
ARMS = ('control', 'S0C', 'instant', 'lag')


def metrics(outputs, scorer):
    if sha(METRIC) != METRIC_SHA:
        raise ValueError('official metric implementation changed')
    spec = importlib.util.spec_from_file_location('official_protocol364', METRIC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with np.load(OLD / 'data/proteus_windows.npz', allow_pickle=False) as z:
        y = z['y']
        metadata = {k: z[k] for k in ('subject', 'session', 'interface')}
    people = np.array([p.rsplit('/', 1)[1] for p in metadata['subject']])
    metadata['stream'] = np.char.add(np.char.add(metadata['subject'], '|'), metadata['session'])
    cells = np.zeros(10, dtype=int)
    values = {a: np.empty((len(ORDERS), 10)) for a in ARMS}
    for oi, order in enumerate(ORDERS):
        for pi, p in enumerate(scorer.PEOPLE):
            mask = people == p
            for a in ARMS:
                result = module.score(y[mask], outputs[a, order][mask].argmax(1), metadata, mask)
                cells[pi] = result['n_cells']
                values[a][oi, pi] = 100 * result['balanced_accuracy_cell_mean']
    comparisons = [('lag', 'instant'), ('lag', 'S0C'), ('lag', 'control'),
                   ('instant', 'S0C'), ('instant', 'control'), ('S0C', 'control')]
    for a, b in comparisons:
        values[f'{a}_minus_{b}'] = values[a] - values[b]
    summaries = {key: dict(official=scorer.cell_weighted_summary(v, ORDERS, cells),
                           people_order=scorer.summarize(v, ORDERS, scorer.PEOPLE))
                 for key, v in values.items()}
    if abs(summaries['control']['official']['mean'] - 76.27933502145758) > 1e-7:
        raise ValueError('exact official deployment anchor changed')
    changes = {f'{a}_minus_{b}': {
        o: int(np.count_nonzero(outputs[a, o].argmax(1) != outputs[b, o].argmax(1)))
        for o in ORDERS} for a, b in comparisons}
    return summaries, changes


def run():
    proof = json.loads((HERE / 'proof.json').read_text())
    if len(proof['streams']) != 24 or len(proof['folds']) != 10:
        raise ValueError('incomplete label-free proof')
    s = load_scorer()
    s.verify_inputs(proof['inputs'], {})
    for n, h in proof['streams'].items():
        if sha(ROOT / 'runs/representation_streams' / n) != h:
            raise ValueError(f'candidate stream changed: {n}')
    outputs = validate(ORDERS, tuple(FAMILIES), proof['streams'])
    if len(outputs) != 40 or any(v.shape != (ROWS, 3) for v in outputs.values()):
        raise ValueError('incomplete guarded B2_pool cells')
    print('OK: every pinned order and frozen zero passed before reading held outcomes', flush=True)
    summaries, changes = metrics(outputs, s)
    with (HERE / 'result.json').open('x') as f:
        json.dump(dict(scope='B2_pool FULL-SYSTEM new-person public DEV; no returning/SEALED estimate',
                       summary=summaries, changes=changes, data_rows=ROWS, orders=ORDERS,
                       proof_sha256=sha(HERE / 'proof.json')), f, indent=2)


if __name__ == '__main__':
    run()
