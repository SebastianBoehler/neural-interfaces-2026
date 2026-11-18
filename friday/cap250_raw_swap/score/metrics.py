"""Official cell-balanced accuracy and paired Scherer effects."""

import numpy as np

from bank import ORDERS, ROWS
from protocol import score


def class_cells(labels, matrix, subject, day, meta, people, archive):
    if labels.shape != (ROWS,) or not np.issubdtype(labels.dtype, np.integer) or not np.isin(labels, (0, 1, 2)).all():
        raise ValueError("invalid Scherer gold after receipt lock")
    classes = np.full((len(matrix), len(people), 2, len(ORDERS), 3), np.nan)
    for i, person in enumerate(people):
        for v, visit in enumerate(("1", "2")):
            mask = (subject == person) & (day == visit)
            if not np.array_equal(np.unique(labels[mask]), [0, 1, 2]):
                raise ValueError(f"missing a gold class in {person} visit {visit}")
            for c in range(3):
                classes[:, i, v, :, c] = 100 * (matrix[:, :, mask & (labels == c)] == c).mean(axis=-1)
    cells = classes.mean(axis=-1)
    full = np.ones(ROWS, dtype=bool)
    for a, arm in enumerate(matrix):
        for j, pred in enumerate(arm):
            official = score(labels, pred, meta, full)
            if official["n_cells"] != 18:
                raise ValueError("official scorer observed a different set of cells")
            np.testing.assert_allclose(100 * official["balanced_accuracy_cell_mean"],
                                       cells[a, :, :, j].mean(), rtol=0, atol=1e-10)
            np.testing.assert_allclose(np.sort(100 * np.asarray(official["cell_scores"])),
                                       np.sort(cells[a, :, :, j].reshape(-1)), rtol=0, atol=1e-10)
    stock = archive["arms"].index("cap")
    np.testing.assert_allclose(cells[0].mean(), archive["acc"][stock], rtol=0, atol=1e-10)
    np.testing.assert_allclose(classes[0].mean(axis=(0, 1, 2)), archive["classes"][stock], rtol=0, atol=1e-10)
    return classes, cells


def boot_draws():
    rng = np.random.default_rng(468)
    people = rng.integers(9, size=(10000, 9, 1, 1))
    visits = rng.integers(2, size=(10000, 9, 2, 1))
    orders = rng.integers(2, size=(10000, 9, 2, 2))
    return people, visits, orders


def interval(values):
    return np.quantile(values, (0.1, 0.9)).tolist()


def timing(stock, candidate):
    ratio = candidate / stock
    median, maximum = float(np.median(ratio)), float(ratio.max())
    return dict(sessions=int(ratio.size), stock_seconds=float(stock.sum()), candidate_seconds=float(candidate.sum()),
                paired_median_ratio=median, max_session_ratio=maximum, total_ratio=float(candidate.sum() / stock.sum()),
                passes_a100_gate=median <= 1.05 and maximum <= 1.10)


def counts(stock, candidate, labels):
    old = stock == labels
    new = candidate == labels
    rescued = int((new & ~old).sum())
    lost = int((old & ~new).sum())
    return dict(changed=int((candidate != stock).sum()), rescued=rescued, lost=lost, net_rescued=rescued - lost)


def contrast(a, matrix, runtime, classes, cells, labels, people, draws):
    p, v, o = draws
    delta = cells[a] - cells[0]
    pooled = interval(delta[p, v, o].mean(axis=(1, 2, 3)))
    by_order = {}
    for j, order in enumerate(ORDERS):
        diff = delta[:, :, j]
        by_order[order] = dict(
            pp=float(diff.mean()), paired80=interval(diff[p[..., 0], v[..., 0]].mean(axis=(1, 2))),
            absolute_pct=dict(stock=float(cells[0, :, :, j].mean()), candidate=float(cells[a, :, :, j].mean())),
            class_pct=classes[a, :, :, j].mean(axis=(0, 1)).tolist(),
            decisions=counts(matrix[0, j], matrix[a, j], labels),
            warm=timing(runtime[0, :, :, j], runtime[a, :, :, j]),
        )
    warm = timing(runtime[0], runtime[a])
    supported = pooled[0] > 0 and all(order["pp"] > 0 for order in by_order.values())
    return dict(pp=float(delta.mean()), paired80=pooled, by_order=by_order,
                people_pp=dict(zip(people, delta.mean(axis=(1, 2)).tolist(), strict=True)),
                class_delta_pp=(classes[a] - classes[0]).mean(axis=(0, 1, 2)).tolist(),
                decisions=counts(matrix[0], matrix[a], labels), warm=warm,
                accuracy_outcome="supports_gain" if supported else "refuted" if pooled[1] <= 0 else "undecided")
