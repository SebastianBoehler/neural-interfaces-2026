"""The conditional penalty must retain class separation and detect within-class shift."""
import importlib.util
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent


def load():
    path = HERE / 'loss.py'
    assert path.is_file(), 'class-conditioned CORAL loss is missing'
    spec = importlib.util.spec_from_file_location('conditional_coral', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def rows(nuisance_scale=1.):
    values, people, classes = [], [], []
    for person in range(2):
        for cls, center in enumerate((-2., 0., 2.)):
            for sign in (-1., 1.):
                values.append([center * (person + 1), sign * .5 * (nuisance_scale if person else 1.)])
                people.append(person); classes.append(cls)
    return (torch.tensor(values, dtype=torch.float64, requires_grad=True),
            torch.tensor(people), torch.tensor(classes))


def test_class_separation_is_not_penalized_as_a_person_covariance_shift():
    module = load(); x, person, label = rows()
    loss, pairs, by_class = module.conditional(x, person, label)
    assert loss.item() == 0 and pairs == 3 and by_class == [1, 1, 1]
    # Unconditional covariances differ because class means separate more for person1.
    assert not torch.equal(torch.cov(x[person == 0].T), torch.cov(x[person == 1].T))


def test_within_class_variance_shift_has_the_known_penalty_and_real_gradient():
    module = load(); x, person, label = rows(nuisance_scale=2.)
    loss, pairs, by_class = module.conditional(x, person, label)
    # Each class: (2.0 - 0.5)^2 / (4 * 2^2) = 0.140625.
    assert loss.item() == pytest.approx(.140625, abs=1e-14)
    assert pairs == 3 and by_class == [1, 1, 1]
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.norm().item() > 0


def test_each_person_class_mean_can_move_without_changing_conditional_penalty():
    module = load(); x, person, label = rows(nuisance_scale=2.)
    original = module.conditional(x, person, label)[0]
    moved = x + torch.column_stack((person * (label + 2.), label * (person + 3.)))
    assert module.conditional(moved, person, label)[0].item() == pytest.approx(original.item(), abs=1e-14)


def test_insufficient_same_class_pairs_are_counted_and_invalid_labels_rejected():
    module = load(); x, person, label = rows()
    one_person = person == 0
    loss, pairs, by_class = module.conditional(x[one_person], person[one_person], label[one_person])
    assert loss.item() == 0 and pairs == 0 and by_class == [0, 0, 0]
    with pytest.raises(ValueError):
        module.conditional(x, person, label + 3)
