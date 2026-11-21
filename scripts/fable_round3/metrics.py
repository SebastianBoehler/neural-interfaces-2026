"""Official cell scores and paired conditional accuracy summaries."""
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from settings import D
from verify_collect import contrast


@dataclass
class Session:
    mode: str
    person: str
    visit: str
    order: int
    y: np.ndarray
    interface: np.ndarray
    tape: dict

    @cached_property
    def masks(self):
        return {str(i): self.interface == i for i in sorted(set(self.interface))}

    @cached_property
    def weights(self):
        weights = np.empty(len(self.y))
        for mask in self.masks.values():
            classes = np.unique(self.y[mask])
            for label in classes:
                hit = mask & (self.y == label)
                weights[hit] = 1 / (len(classes) * hit.sum())
        np.testing.assert_allclose(weights.sum(), len(self.masks), rtol=0, atol=1e-10)
        return weights

    @cached_property
    def changed(self):
        return self.tape["pre_em"].argmax(1) != self.tape["hard"]

    @cached_property
    def age(self):
        return np.arange(1, len(self.y) + 1)

    def correct(self, label):
        return np.asarray(label) == self.y

    def cells(self, label):
        correct = self.correct(label)
        return {(self.mode, self.order, self.person, self.visit, i): 100 * sum(self.weights[m] * correct[m])
                for i, m in self.masks.items()}


def official_cells(cases, predict):
    scores = {}
    for case in cases:
        cells = case.cells(predict(case))
        if cells.keys() & scores.keys():
            raise ValueError("duplicate official cell")
        scores.update(cells)
    return scores


def paired(cases, a, b, orders):
    aa, bb = (official_cells(cases, fn) for fn in (a, b))
    for order in orders:
        assert sum(k[:2] == ("new", order) for k in aa) == 28
        assert sum(k[:2] == ("ret", order) for k in aa) == 8
    return contrast(aa, bb, orders)


def summarize(mask_a, label_a, cases, orders, mask_b=None, label_b=None):
    """Use full-cell weights. Renormalize by stratum mass in each bootstrap draw."""
    out, draws_all = {}, []
    seed = np.random.default_rng(20261121).integers(len(orders), size=(20_000, len(orders)))
    people_rng = np.random.default_rng(71120)
    for mode, people in (("new", D.PEOPLE), ("ret", D.RETURNING)):
        bank = np.zeros((len(orders), len(people), 2, 2))
        selectors = [(mask_a, label_a)] if mask_b is None else [(mask_a, label_a), (mask_b, label_b)]
        for case in cases:
            if case.mode != mode:
                continue
            i, j = orders.index(case.order), people.index(case.person)
            for arm, (select, predict) in enumerate(selectors):
                mask = select(case)
                w = case.weights[mask]
                bank[i, j, arm] += [w @ case.correct(predict(case))[mask], w.sum()]
        pick = people_rng.integers(len(people), size=(len(seed), len(people)))
        sampled = bank[seed[:, :, None], pick[:, None, :]].sum((1, 2))

        def measure(values):
            correct, mass = np.moveaxis(values, -1, 0)
            scores = np.divide(100 * correct, mass, out=np.full_like(correct, np.nan), where=mass > 0)
            return scores[..., 0] if mask_b is None else scores[..., 0] - scores[..., 1]

        point = float(measure(bank.sum((0, 1))))
        if not np.isfinite(point):
            raise ValueError(f"empty stratum in {mode}")
        boot = measure(sampled)
        out["new" if mode == "new" else "returning"] = dict(mean=point, ci80=np.nanquantile(boot, [0.1, 0.9]).tolist(),
                                                          omitted_bootstrap=int(np.isnan(boot).sum()))
        draws_all.append(boot)
    out["j"] = (out["new"]["mean"] + out["returning"]["mean"]) / 2
    out["j_ci80"] = np.nanquantile((draws_all[0] + draws_all[1]) / 2, [0.1, 0.9]).tolist()
    return out


def column(name):
    return lambda case: case.tape[name]


def arm(name):
    return lambda case: case.tape[name].argmax(1)
