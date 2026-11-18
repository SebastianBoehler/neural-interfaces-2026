"""Signed contributions to balanced accuracy, with full-class denominators."""
import numpy as np


def position_bins(rows, order, streams):
    if len(set(rows)) != len(rows) or not np.array_equal(np.sort(order), np.sort(rows)):
        raise ValueError("order is not a bijection of row identifiers")
    lookup = dict(zip(rows, range(len(rows))))
    indices = np.asarray([lookup[r] for r in order])
    bins = np.empty(len(rows), int)
    for stream in np.unique(streams):
        sequence = indices[streams[indices] == stream]
        bins[sequence] = np.digitize(np.arange(1, len(sequence) + 1), [51, 91])
    return bins


def decompose(labels, cells, owners, people, bins, old, new):
    if not all(np.isin(a, [0, 1, 2]).all() and a.shape == labels.shape for a in (old, new, labels)):
        raise ValueError("invalid decisions")
    person_stats, weights = [], []
    for person in people:
        cell_stats = []
        for cell in np.unique(cells[owners == person]):
            mask = cells == cell
            if not np.all(owners[mask] == person):
                raise ValueError("mixed official cell")
            stats = np.zeros((3, 3, 3))
            for cls in range(3):
                m = mask & (labels == cls)
                if not m.any():
                    raise ValueError("missing class in official cell")
                for measure, select in enumerate((m, m & (old != cls) & (new == cls),
                                                  m & (old == cls) & (new != cls))):
                    stats[measure, cls] = 100 / 3 * np.bincount(bins[select], minlength=3) / m.sum()
            cell_stats.append(stats)
        person_stats.append(np.mean(cell_stats, axis=0))
        weights.append(len(cell_stats))
    return np.array(person_stats), np.array(weights)


def contrasts(stats):
    exposure, corrections, regressions = np.moveaxis(stats, -3, 0)
    net = corrections - regressions
    late = net[..., 2].sum(-1) - exposure[..., 2].sum(-1) * net.sum((-2, -1)) / 100
    cls = net.sum(-1) * 3
    return np.stack((late, cls[..., 2] - cls[..., :2].mean(-1)), axis=-1)


def describe(stats):
    exposure, corrections, regressions = stats
    net = corrections - regressions
    return dict(delta_pp=float(net.sum()), late_excess_pp=float(contrasts(stats)[0]),
                word_selectivity_pp=float(contrasts(stats)[1]), class_delta_pp=(3*net.sum(-1)).tolist(),
                exposure_pct=exposure.tolist(), corrections_pp=corrections.tolist(),
                regressions_pp=regressions.tolist())


def summarize(matrix, weights, names, groups):
    # Shared order samples preserve package and participant pairing within a draw.
    schemes = {s: matrix[g].mean(0) for s, g in groups.items()}
    people = np.mean(list(schemes.values()), axis=0)
    point = np.average(people, axis=0, weights=weights)
    result = describe(point)
    rng = np.random.default_rng(427)
    n = len(weights)
    draws = rng.integers(n, size=(6000, n))
    order_draws = [rng.choice(g, (len(draws), len(g))) for g in groups.values()]
    cis = []
    for start in range(0, len(draws), 100):
        batch = slice(start, start+100)
        p = draws[batch]
        orders = np.mean([matrix[g[batch]].mean(1) for g in order_draws], axis=0)
        sampled = orders[np.arange(len(p))[:, None], p]
        cis.extend(contrasts(np.sum(sampled * weights[p, None, None, None], axis=1)
                             / weights[p].sum(1)[:, None, None, None]))
    result.update(interval80=np.quantile(cis, [.1, .9], axis=0).T.tolist(),
                  people={str(k): describe(v) for k, v in zip(names, people)},
                  schemes={s: describe(np.average(v, axis=0, weights=weights)) for s, v in schemes.items()})
    return result, point
