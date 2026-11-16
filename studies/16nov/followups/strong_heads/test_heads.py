import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from heads import Structured, fit_scale, ridge_fit, ridge_predict, RandomMap  # noqa: E402
from split import source_partition  # noqa: E402
from train import fit_structured, fit_ridge_grid, refit_structured, refit_ridge  # noqa: E402
from export import structured_export, ridge_export  # noqa: E402
from geometry import source_prior, sessions_in_order  # noqa: E402


def test_structured_matched_head_has_active_nonlinear_path():
    torch.manual_seed(3)
    linear = Structured(False)
    nonlinear = Structured(True)
    nonlinear.load_state_dict(linear.state_dict())
    x = torch.randn(3, 44239)
    out_linear = linear(x)
    out_nonlinear = nonlinear(x)
    assert out_linear.shape == (3, 3)
    assert not torch.allclose(out_linear, out_nonlinear)
    out_nonlinear.square().mean().backward()
    assert nonlinear.block.weight.grad.norm().item() > 0
    assert nonlinear.slow.weight.grad.norm().item() > 0


def test_scale_rejects_nonfinite_input():
    with pytest.raises(ValueError, match='finite'):
        fit_scale(np.array([[1.0, np.nan], [2.0, 4.0]]))


def test_ridge_fit_uses_unpenalized_intercept():
    x = np.array([[-1.0], [0.0], [1.0], [2.0]])
    y = np.array([0, 0, 0, 1])
    mean, scale = fit_scale(x)
    w = ridge_fit((x - mean) / scale, y, 0.001)
    p = ridge_predict((x - mean) / scale, w, 1.0)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert p.shape == (4, 3)
    assert np.allclose(w[-1], np.bincount(y, minlength=3) / 4)


def test_random_map_reuses_fitted_source_statistics():
    x = np.random.default_rng(3).normal(size=(32, 9))
    for arm, width in [('linear', 12), ('quadratic', 24), ('fourier', 12)]:
        mapper = RandomMap.fit(x, arm, 7, width=12)
        first = mapper.transform(x[:2])
        assert first.shape == (2, width)
        assert np.array_equal(first, mapper.transform(x[:2]))
        assert not np.allclose(first, mapper.transform(x[2:4]))


def test_inner_and_outer_people_are_both_excluded():
    people = np.repeat(np.arange(10), 2)
    train, inner, outer = source_partition(people, outer_person=2, seed=0)
    assert len(set(people[train])) == 8
    assert len(set(people[inner])) == len(set(people[outer])) == 1
    assert not set(people[train]) & (set(people[inner]) | set(people[outer]))
    assert set(people[outer]) == {2}


def test_head_selection_uses_source_scaler_and_registered_choices():
    rng = np.random.default_rng(4)
    train = rng.normal(size=(18, 44239)).astype('float32')
    valid = rng.normal(size=(6, 44239)).astype('float32')
    y = np.tile(np.arange(3), 6)
    model, scale, selected, losses = fit_structured(train, y, valid, y[:6], False, 0,
                                                     epochs=(1, 2), batch=9)
    assert selected in (1, 2) and len(losses) == 2
    assert np.allclose(scale[0], train.mean(axis=0), atol=1e-6)
    assert torch.isfinite(model(torch.as_tensor((valid - scale[0]) / scale[1], dtype=torch.float32))).all()


def test_ridge_grid_records_only_registered_knobs():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(30, 8))
    y = np.tile(np.arange(3), 10)
    model = fit_ridge_grid(x, y, x[:6], y[:6], 'fourier', 1, width=8)
    assert model['strength'] in (1e-4, 1e-3)
    assert model['temperature'] in (1, 4)
    assert len(model['validation']) == 4


def test_scripted_structured_checkpoint_matches_reference(tmp_path):
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(4, 44239)).astype('float32')
    mean, scale = fit_scale(raw)
    model = Structured(True).eval()
    saved = tmp_path / 'structured.pt'
    structured_export(model, mean, scale, raw[:1], saved)
    with torch.no_grad():
        expected = model(torch.as_tensor((raw - mean) / scale, dtype=torch.float32)).softmax(1).numpy()
        actual = torch.jit.load(str(saved))(torch.from_numpy(raw.astype('float64'))).numpy()
    assert np.allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_scripted_ridge_checkpoint_matches_reference(tmp_path):
    rng = np.random.default_rng(5)
    x = rng.normal(size=(30, 8)).astype('float32')
    y = np.tile(np.arange(3), 10)
    model = fit_ridge_grid(x, y, x[:6], y[:6], 'quadratic', 0, width=8)
    z = ((x[:4] - model['mean']) / model['scale']).astype('float32')
    expected = ridge_predict(model['mapper'].transform(z), model['weights'], model['temperature'])
    saved = tmp_path / 'ridge.pt'
    ridge_export(model, x[:1], saved)
    actual = torch.jit.load(str(saved))(torch.from_numpy(x[:4].astype('float64'))).numpy()
    assert np.allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_inner_prior_never_uses_either_held_person():
    priors = np.arange(10.0).reshape(10, 1, 1)
    counts = np.arange(1, 11)
    result = source_prior(priors, counts, [2, 5])
    keep = [i for i in range(10) if i not in (2, 5)]
    assert np.allclose(result, np.average(priors[keep], axis=0, weights=counts[keep]))
    priors[[2, 5]] = 1e10
    assert np.allclose(result, source_prior(priors, counts, [2, 5]))


def test_pinned_order_groups_each_session_in_delivery_order():
    permutation = np.array([3, 0, 2, 1])
    sessions = np.array(['a', 'a', 'b', 'b'])
    result = sessions_in_order(permutation, np.array([0, 1, 2, 3]), sessions)
    assert [rows.tolist() for rows in result] == [[3, 2], [0, 1]]
    with pytest.raises(ValueError, match='permutation'):
        sessions_in_order(np.array([0, 1, 1, 3]), np.arange(4), sessions)


def test_fixed_epoch_refit_replays_selected_training():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(12, 44239)).astype('float32')
    y = np.tile(np.arange(3), 4)
    selected, scale, epoch, _ = fit_structured(x, y, x[:6], y[:6], True, 0, epochs=(1, 2), batch=6)
    refit, refit_scale = refit_structured(x, y, True, 0, epoch, batch=6)
    assert np.array_equal(scale[0], refit_scale[0])
    for a, b in zip(selected.parameters(), refit.parameters()):
        assert torch.equal(a, b)


def test_fixed_ridge_refit_replays_same_source_map():
    x = np.random.default_rng(8).normal(size=(30, 7)).astype('float32')
    y = np.tile(np.arange(3), 10)
    chosen = fit_ridge_grid(x, y, x[:6], y[:6], 'linear', 0, width=8)
    refit = refit_ridge(x, y, 'linear', 0, chosen['strength'], chosen['temperature'], width=8)
    assert np.array_equal(chosen['mapper'].projection, refit['mapper'].projection)
    assert np.array_equal(chosen['weights'], refit['weights'])
