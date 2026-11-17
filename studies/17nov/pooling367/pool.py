"""Pool the current probabilities from two complete, independently replayed branches."""
import json
from pathlib import Path
import sys

import numpy as np

PARENT = Path('/mnt/lustre/work/gehler/gfh527/ni26_selector363_v1_20261117')
sys.path.insert(0, str(PARENT))
from pilot import (CHAIN, OWNER, chain_guard, official, pin, probability_rows, read_labels,
                   require, sha, verify_sources)

ARMS = ('base', 'JS', 'arithmetic', 'geometric')


def pool(scores, domain):
    logs = np.where(domain[..., None], np.log(np.clip(scores, 1e-30, 1.)), scores)
    weights = np.exp(logs - logs.max(axis=-1, keepdims=True))
    probabilities = weights / weights.sum(axis=-1, keepdims=True)
    labels = scores.argmax(axis=-1)
    require(np.array_equal(labels, probabilities.argmax(axis=-1)), 'normalization changed endpoints')
    arithmetic = probabilities.sum(axis=0).argmax(axis=-1)
    geometric = logs.sum(axis=0).argmax(axis=-1)
    agree = labels[0] == labels[1]
    require(all(np.array_equal(p[agree], labels[0][agree]) for p in (arithmetic, geometric)),
            'pool changed an agreed label')
    return (*labels, arithmetic, geometric)


def smoke():
    scores = np.array([[[.55, .40, .05], [.34, .33, .33]],
                       [[.01, .59, .40], [.34, .32, .34]]])
    scores[:, 1] = np.log(scores[:, 1])
    a, b, arithmetic, geometric = pool(scores, np.array([True, False]))
    require(a.tolist() == [0, 0] and b.tolist() == [1, 0]
            and arithmetic.tolist() == geometric.tolist() == [1, 0], 'mixed-domain smoke failed')
    print('POOL_SMOKE_OK', flush=True)


def evaluate(meta, people, scorer, predictions):
    labels = read_labels(np.arange(len(people)), len(people))
    metric = official()
    values = np.empty((len(ARMS), len(scorer.ORDERS), len(scorer.BASES), len(scorer.PEOPLE)))
    counts = np.empty(len(scorer.PEOPLE), dtype=int)
    for i, person in enumerate(scorer.PEOPLE):
        keep = people == person
        for a, o, b in np.ndindex(values.shape[:3]):
            row = metric(labels[keep], predictions[a][o, b, keep], meta, keep)
            values[a, o, b, i] = 100 * row['balanced_accuracy_cell_mean']
            counts[i] = row['n_cells']
    out = {}
    for name, bases in (('all_bases', slice(None)), ('B2_pool', slice(1))):
        means = values[:, :, bases].mean(axis=2)
        report = {arm: scorer.cell_weighted_summary(means[i], scorer.ORDERS, counts)
                  for i, arm in enumerate(ARMS)}
        for i, arm in enumerate(ARMS[2:], start=2):
            for j, other in enumerate(ARMS[:i]):
                delta = means[i] - means[j]
                report[f'{arm}_minus_{other}'] = {
                    'official': scorer.cell_weighted_summary(delta, scorer.ORDERS, counts),
                    'people_order': scorer.summarize(delta, scorer.ORDERS, scorer.PEOPLE)}
        out[name] = report
    require(abs(out['all_bases']['base']['mean'] - 76.21767294925993) < 1e-8
            and abs(out['B2_pool']['JS']['mean'] - 76.55173151211109) < 1e-8,
            'frozen score endpoints changed')
    return out


def main():
    smoke()
    if sys.argv[1:] == ['--smoke']:
        return
    require(not sys.argv[1:], 'unknown pool arguments')
    verify_sources()
    pin(OWNER/'source.sha256', '9e889431c92f4759c79e0724446d19fc9503ac5b726464f73f9ed22b548f9ccd')
    with np.load(CHAIN/'data/proteus_windows.npz', allow_pickle=False) as z:
        meta = {k: z[k] for k in ('subject', 'session', 'interface', 'task', 'run')}
    meta['person'] = np.array([p.rsplit('/', 1)[-1] for p in meta['subject']])
    meta['stream'] = np.char.add(np.char.add(meta['subject'], '|'), meta['session'])
    meta['group'] = np.char.add(np.char.add(meta['task'], '|'), meta['run'].astype(str))
    scorer, scores, people, _ = chain_guard(meta)
    masks = probability_rows(meta, scores, scorer)
    domain = np.stack([masks[o] for o in scorer.ORDERS])[:, None, :]
    scores = np.stack([[[scores[a, o, b] for b in scorer.BASES] for o in scorer.ORDERS]
                       for a in ('control', 'JS')])
    require(scores.shape == (2, 8, 6, 2655, 3) and np.isfinite(scores).all()
            and np.array_equal((scores >= 0).all(-1), np.broadcast_to(domain, scores.shape[:-1])),
            'incomplete branch scores or inconsistent score domains')
    predictions = pool(scores, domain)
    base, js, *pooled = predictions
    require(int((base != js).sum()) == 4021 and int((base[:, 0] != js[:, 0]).sum()) == 682
            and all(np.any(p != js) for p in pooled), 'frozen disagreements or active pool changed')
    np.savez_compressed('predictions.npz', **dict(zip(ARMS, predictions)))
    proof = {'labels_opened': False, 'source_manifest': sha(Path('source.sha256')),
             'predictions': sha(Path('predictions.npz')),
             'changed_vs_JS': {a: int((p != js).sum()) for a, p in zip(ARMS[2:], pooled)},
             'third_label': {a: int(((p != base) & (p != js)).sum()) for a, p in zip(ARMS[2:], pooled)}}
    with Path('before_labels.json').open('x') as f:
        json.dump(proof, f, indent=2)
    result = evaluate(meta, people, scorer, predictions)
    result['proof_sha256'] = sha(Path('before_labels.json'))
    with Path('result.json').open('x') as f:
        json.dump(result, f, indent=2)
    print('POOL_SCORED', flush=True)


if __name__ == '__main__':
    main()
