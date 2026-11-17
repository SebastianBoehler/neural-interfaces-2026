"""Collect all six split outputs before opening labels and form paired contrasts."""
import json
from pathlib import Path

import numpy as np
from rotation import ARMS, COHORT, PAIRS, checksum, inputs, source
from report import aggregate


def main():
    outputs, manifest = [], []
    for i, pair in enumerate(PAIRS):
        root = Path('fold'+str(i))
        card = json.loads((root/'proof.json').read_text())
        test = tuple(p for p in COHORT if p not in pair)
        source.require(tuple(card['train']) == pair and tuple(card['test']) == test
                       and card['scoring_labels_open'] is False and card['sha'] == checksum(root/'members.npz'),
                       'split scope/identity differs')
        with np.load(root/'members.npz') as z:
            source.require(set(z.files) == {p+'_'+a for p in test for a in ARMS}, 'split matrix incomplete')
            outputs.append({k:z[k] for k in z.files})
        manifest.append(card['sha'])
    Path('before_matrix_labels.json').write_text(json.dumps(manifest))
    rows, meta, people, base = inputs()
    scorer = source.official()
    source.pin(Path(scorer.__file__), source.METRIC_SHA)
    order = source.ORDERS
    sums, counts = {}, np.zeros(len(COHORT), int)
    folds = {}
    for pair, output in zip(PAIRS, outputs):
        test = tuple(p for p in COHORT if p not in pair)
        # The original score computes six matched intervals. Skip them here: split-level
        # estimands are descriptive, while uncertainty belongs to the four-person matrix.
        for p in test:
            i = COHORT.index(p)
            mask = people == p
            y = source.labels(rows[p],len(people))
            def score(pred):
                row = scorer.score(y,pred,meta,mask)
                counts[i] = row['n_cells']
                return 100 * row['balanced_accuracy_cell_mean']
            local = {}
            for j, b in enumerate(base[p]):
                pred = b.argmax(-1)
                obs = {'base':score(pred)}
                for a in ARMS:
                    q = output[p+'_'+a]
                    final = (.75*b+.25*q).argmax(-1)
                    obs[a] = score(final)
                    obs[a+'_member'] = score(q.argmax(-1))
                    obs[a+'_rescue'] = score(np.where(pred != y, final,pred)) - obs['base']
                    obs[a+'_harm'] = obs['base'] - score(np.where(pred == y, final,pred))
                    source.require(np.isclose(obs[a]-obs['base'],obs[a+'_rescue']-obs[a+'_harm']), 'net attribution')
                for k,v in obs.items():
                    if k not in sums:
                        sums[k] = np.zeros((len(COHORT),len(order)))
                    sums[k][i,j] += v/3
                    local.setdefault(k,[]).append(v)
            folds['+'.join(pair)+'->'+p] = local
    summary, contrasts = {}, {}
    for k, v in sums.items():
        est, by = aggregate(v,counts,order)
        summary[k] = dict(official=est[0], people=est[1], by_person=dict(zip(COHORT,by)))
    for a,b in [('error',k) for k in ('ordinary','confidence','shuffled','base')] + [('error_member','ordinary_member')]:
        d = sums[a]-sums[b]
        rng = np.random.default_rng(372)
        est, by = aggregate(d,counts,order)
        ci = np.percentile([aggregate(d,counts,order,rng)[0] for _ in range(4000)], [10,90],axis=0)
        contrasts[a+'-'+b] = dict(official=est[0], official80=ci[:,0].tolist(), people=est[1],
                                  people80=ci[:,1].tolist(), by_person=dict(zip(COHORT,by)))
    Path('result.json').write_text(json.dumps(dict(summary=summary,contrasts=contrasts,folds=folds,
        manifests=manifest,counts=counts.tolist(),orders=order,source_sha=checksum(Path('source.sha256'))),indent=2))


if __name__ == '__main__':
    main()
