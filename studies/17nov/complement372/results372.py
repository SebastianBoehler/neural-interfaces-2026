"""Score frozen predictions and attribute each fused decision change to rescue or harm."""
import numpy as np
from report import aggregate


def collect(scorer, labels, meta, people, rows, cohort, orders, arms, output, base):
    kinds = ['base'] + [a+s for a in arms for s in ('', '_member', '_rescue', '_harm')]
    values = {key: np.empty((len(cohort), len(orders))) for key in kinds}
    counts = np.zeros(len(cohort), int)
    for i, person in enumerate(cohort):
        y = labels(rows[person], len(people))
        mask = people == person
        def score(pred):
            row = scorer.score(y, pred, meta, mask)
            counts[i] = row['n_cells']
            return row['balanced_accuracy_cell_mean'] * 100
        for j, order in enumerate(orders):
            b = base[person][j]
            original = b.argmax(-1)
            values['base'][i, j] = score(original)
            for arm in arms:
                q = output[person+'_'+arm]
                decision = (.75*b + .25*q).argmax(-1)
                values[arm][i, j] = score(decision)
                values[arm+'_member'][i, j] = score(q.argmax(-1))
                rescued = (original != y) & (decision == y)
                harmed = (original == y) & (decision != y)
                values[arm+'_rescue'][i, j] = score(np.where(rescued, decision, original)) - score(original)
                values[arm+'_harm'][i, j] = score(original) - score(np.where(harmed, decision, original))
                if not np.isclose(values[arm][i,j] - score(original),
                                  values[arm+'_rescue'][i,j] - values[arm+'_harm'][i,j]):
                    raise ValueError('rescue/harm decomposition failed')
    summary = {}
    for key, val in values.items():
        pair, by_person = aggregate(val, counts, orders)
        summary[key] = dict(official=pair[0], people_equal=pair[1], by_person=dict(zip(cohort, by_person)))
    contrasts = {}
    for a, b in [('error', 'ordinary'), ('error', 'confidence'), ('error_member', 'ordinary_member'),
                 ('error', 'base'), ('ordinary','base'), ('confidence','base')]:
        delta = values[a] - values[b]
        rng = np.random.default_rng(372)
        est, per = aggregate(delta, counts, orders)
        boot = [aggregate(delta, counts, orders, rng)[0] for _ in range(4000)]
        ci = np.percentile(boot, [10,90], axis=0)
        contrasts[a+'-'+b] = dict(official=est[0], official80=ci[:,0].tolist(), people=est[1],
                                  people80=ci[:,1].tolist(), by_person=dict(zip(cohort, per)))
    return dict(summary=summary, contrasts=contrasts, orders=orders, cell_counts=counts.tolist(),
                by_order={k:v.tolist() for k,v in values.items()}, cohort=cohort)
