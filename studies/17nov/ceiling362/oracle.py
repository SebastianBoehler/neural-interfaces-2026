"""Paired label-assisted ceiling for the two complete archived chain branches."""
import json
import sys
from pathlib import Path

import numpy as np

OWNER = Path('/mnt/lustre/work/gehler/gfh527/ni26_soft_csp361_v1_20261117')
sys.path.insert(0, str(OWNER))
from assets import CHAIN, RET, RET_REPORT, chain_guard, pin, require, sha, verify_sources
from measure import official


def oracle(base, extra, gold):
    return np.where(extra == gold, extra, base)


def metric_view(metric, labels, meta, people, a, b):
    ceiling = oracle(a, b, labels)
    require(np.array_equal(ceiling == labels, (a == labels) | (b == labels)), 'oracle guard')
    by_person, counts = [], []
    for person in sorted(set(people)):
        mask = people == person
        complete_cells(labels, meta, mask)
        scores = [metric(labels[mask], arm[mask], meta, mask) for arm in (a, b, ceiling)]
        by_person.append([100 * item['balanced_accuracy_cell_mean'] for item in scores])
        counts.append(scores[0]['n_cells'])
    return np.array(by_person), np.array(counts)


def complete_cells(labels, meta, mask):
    for cell in set(zip(meta['stream'][mask], meta['interface'][mask])):
        in_cell = mask & (meta['stream'] == cell[0]) & (meta['interface'] == cell[1])
        require(set(labels[in_cell]) == {0, 1, 2}, 'non-complete cell requires different oracle')


def validate(meta):
    pin(OWNER/'source.sha256', '9e889431c92f4759c79e0724446d19fc9503ac5b726464f73f9ed22b548f9ccd')
    verify_sources()
    meta['person'] = np.array([p.rsplit('/', 1)[-1] for p in meta['subject']])
    meta['stream'] = np.char.add(np.char.add(meta['subject'], '|'), meta['session'])
    scorer, scores, people, _ = chain_guard(meta)
    require(scorer.BASES[0] == 'B2_pool', 'primary frozen base changed')
    pairs = np.stack([[[scores[a, o, b].argmax(1) for b in scorer.BASES] for o in scorer.ORDERS]
                      for a in ('control', 'JS')])
    require(pairs.shape == (2, 8, 6, len(people)), 'incomplete actual paired full chains')
    return scorer, pairs, people


def new_people(scorer, metric, labels, meta, people, pairs):
    values = np.empty((8, 6, len(scorer.PEOPLE), 3))
    for oi in range(8):
        for bi in range(6):
            values[oi, bi], counts = metric_view(metric, labels, meta, people, *pairs[:, oi, bi])
    disagree = pairs[0] != pairs[1]
    result = {}
    for key, which in [('all_bases', slice(None)), ('B2_pool', slice(0, 1))]:
        means = values[:, which].mean(1)
        pairs_report = {a: scorer.cell_weighted_summary(means[:, :, i], scorer.ORDERS, counts)
                        for i, a in enumerate(('base', 'JS', 'oracle'))}
        for index, arm in enumerate(('base', 'JS')):
            delta = means[:, :, 2] - means[:, :, index]
            pairs_report['oracle_minus_'+arm] = dict(
                official=scorer.cell_weighted_summary(delta, scorer.ORDERS, counts),
                people_order=scorer.summarize(delta, scorer.ORDERS, scorer.PEOPLE))
        ds = disagree[:, which]
        pairs_report['disagreement'] = dict(count=int(ds.sum()), compared=int(ds.size),
            unique_rows=int(ds.any((0, 1)).sum()), per_order=dict(zip(scorer.ORDERS, ds.sum((1, 2)).tolist())))
        result[key] = pairs_report
    return result


def returning(scorer, metric, labels, meta):
    pin(RET_REPORT, 'd0532701c0b88a4acd8fd8bd3d6b7b98469ef1d6bae0758b34ce690d50313f04')
    parent = json.loads(RET_REPORT.read_text())
    data = np.empty((2, 8, 3))
    disagreements = np.zeros((2, 8), int)
    for index, owned in enumerate(parent['complete_cells']):
        person = ('P04', 'P10')[index//8]
        folder = RET/f'cell_{index}'
        pin(folder/'proof.json', owned['proof_sha256'])
        pin(folder/'predictions.npz', owned['predictions_sha256'])
        mask = (meta['person'] == person) & (meta['session'] != '1')
        with np.load(folder/'predictions.npz', allow_pickle=False) as z:
            base, js = z['base_labels'], z['js_labels']
            require(np.array_equal(z['index'], np.flatnonzero(mask))
                    and np.array_equal(base, z['off_labels']) and owned['order'] == scorer.ORDERS[index%8],
                    'returning row/control identity')
        labels_at = labels[mask]
        complete_cells(labels, meta, mask)
        n = oracle(base, js, labels_at)
        data[index//8, index%8] = [100 * metric(labels_at, arm, meta, mask)['balanced_accuracy_cell_mean']
                                  for arm in (base, js, n)]
        disagreements[index//8, index%8] = np.sum(base != js)
    require(np.allclose(data[:, :, 0].mean(1),
                        [parent['person_metrics_percent'][p]['base'] for p in ('P04', 'P10')], rtol=0, atol=1e-8),
            'returning official score changed')
    return dict(status='conditional returning oracle, retained production priors',
                p04=dict(zip(('base','JS','oracle'), data[0].mean(0).tolist())),
                p10=dict(zip(('base','JS','oracle'), data[1].mean(0).tolist())),
                returning_weighted_oracle_minus_base=float((data[:, :, 2]-data[:, :, 0]).mean(1) @ [.2,.8]),
                disagreements=disagreements.sum(1).tolist())


def timing():
    pin(Path('warm_pairs.json'), '275c97327c443117a25537dd5e00191fe7a618036b854d58fe052ddc2f7c5488')
    timing = json.loads(Path('warm_pairs.json').read_text())
    require(int(timing['job']) == 3117650 and len(timing['rows']) == 12, 'observed warm pair scope changed')
    out = {}
    for size in (190, 378, 800):
        values = np.array([[r['base_seconds'], r['js_seconds']] for r in timing['rows'] if r['windows'] == size])
        out[str(size)] = dict(base_median_s=float(np.median(values[:,0])), js_median_s=float(np.median(values[:,1])),
                              independent_pair_sum_median_s=float(np.median(values.sum(1))))
    return out


def main():
    with np.load(CHAIN/'data/proteus_windows.npz', allow_pickle=False) as z:
        meta = {k: z[k] for k in ('subject', 'session', 'interface')}
    scorer, pairs, people = validate(meta)
    metric = official()
    with np.load(CHAIN/'data/proteus_windows.npz', allow_pickle=False) as z:
        labels = z['y']
    result = dict(new=new_people(scorer, metric, labels, meta, people, pairs),
                  returning=returning(scorer, metric, labels, meta), pair_proxy=timing(),
                  source_sha256=sha(Path('source.sha256')),
                  interpretation='label-assisted upper bound; fixed archived full branches only')
    with Path('result.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result['new']['B2_pool']), flush=True)


if __name__ == '__main__':
    main()
