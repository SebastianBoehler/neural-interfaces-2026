"""Official cell means and paired bootstrap over people and within-person orders."""
import numpy as np


def aggregate(values, counts, orders, rng=None):
    groups = [[i for i, order in enumerate(orders) if order.startswith(scheme)]
              for scheme in ('within', 'full', 'runs')]
    people = np.arange(len(values)) if rng is None else rng.integers(len(values), size=len(values))
    person = np.array([np.mean([values[p, group if rng is None else rng.choice(group, len(group))].mean()
                               for group in groups]) for p in people])
    return np.array([np.average(person, weights=counts[people]), person.mean()]), person


def score(scorer, labels, metadata, people, cohort, orders, outputs):
    arms = ('A6', 'current', 'prefix')
    scores = {arm: np.empty((len(cohort), len(orders))) for arm in arms}
    counts = np.zeros(len(cohort), dtype=int)
    for person_idx, person in enumerate(cohort):
        mask = people == person
        y = labels(np.flatnonzero(mask), len(people))
        for order_idx, order in enumerate(orders):
            for arm in arms:
                predicted = outputs[person, order, arm].argmax(1)
                row = scorer.score(y, predicted, metadata, mask)
                scores[arm][person_idx, order_idx] = 100 * row['balanced_accuracy_cell_mean']
                if counts[person_idx] not in (0, row['n_cells']):
                    raise ValueError('official cell count changed between arms or orders')
                counts[person_idx] = row['n_cells']
    assert (counts > 0).all()
    summary, by_person, contrasts = {}, {}, {}
    for arm in arms:
        absolute, person = aggregate(scores[arm], counts, orders)
        summary[arm] = dict(official=float(absolute[0]), people_equal=float(absolute[1]))
        by_person[arm] = {p: dict(scheme_equal=float(person[i]),
                                  by_order=dict(zip(orders, scores[arm][i].tolist())))
                          for i, p in enumerate(cohort)}
    for a, b in [('prefix', 'current'), ('prefix', 'A6'), ('current', 'A6')]:
        delta = scores[a] - scores[b]
        means, person = aggregate(delta, counts, orders)
        rng = np.random.default_rng(359)
        boot = np.array([aggregate(delta, counts, orders, rng)[0] for _ in range(4000)])
        ci = np.percentile(boot, (10, 90), axis=0)
        contrasts[a+'-'+b] = dict(official_pp=float(means[0]), official80=ci[:, 0].tolist(),
            people_equal_pp=float(means[1]), people_equal80=ci[:, 1].tolist(),
            by_person=dict(zip(cohort, person.tolist())), people_up=int(np.sum(person > 0)))
    return dict(summary=summary, by_person=by_person, contrasts=contrasts,
                cell_counts=dict(zip(cohort, counts.tolist())))
