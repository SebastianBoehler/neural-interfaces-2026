"""Synthetic checks of causality, reset, identity and anchored updates."""
import numpy as np
import torch
from model import Head, features
from stream import MAX_DELTA, predict_session

torch.set_num_threads(1)


def test_window_features_do_not_read_other_windows():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(3, 41, 480)).astype(np.float32)
    a = features(x)
    x[1:] *= 100
    assert np.array_equal(a[0], features(x)[0])
    assert a.shape == (3, 246) and np.isfinite(a).all()


def test_stream_reset_causality_and_anchor():
    torch.manual_seed(2)
    m = Head().eval()
    with torch.no_grad():
        m.out.weight.mul_(0.15)
        m.out.bias.copy_(torch.tensor([4., 0., 0.]))
    before = {k: v.detach().clone() for k, v in m.state_dict().items()}
    x = np.random.default_rng(1).normal(size=(65, 246)).astype(np.float32)
    fixed, disabled, _ = predict_session(m, x, 8, update=False)
    ref, live, diag = predict_session(m, x, 8, update=True)
    assert np.array_equal(fixed, disabled) and np.array_equal(fixed, ref)
    assert np.array_equal(fixed[:20], live[:20])
    assert sum(diag['selected']) > 0 and 0 < diag['delta'] <= MAX_DELTA + 1e-6
    assert not np.array_equal(fixed[20:], live[20:])
    changed = x.copy(); changed[50:] = 40
    _, changed_live, _ = predict_session(m, changed, 8, update=True)
    assert np.array_equal(live[:50], changed_live[:50])
    unrelated = np.random.default_rng(5).normal(size=(43, 246)).astype(np.float32)
    predict_session(m, unrelated, 123, update=True)  # another stream runs between two visits
    _, rerun, _ = predict_session(m, x, 8, update=True)
    assert np.array_equal(live, rerun), 'state crossed the session boundary'
    assert all(torch.equal(v, before[k]) for k, v in m.state_dict().items()), 'population model mutated'


def test_no_confident_teacher_gives_a_noop():
    m = Head().eval()
    with torch.no_grad():
        for p in m.parameters(): p.zero_()
    x = np.ones((45, 246), dtype=np.float32)
    fixed, live, diag = predict_session(m, x, update=True)
    assert np.array_equal(fixed, live) and sum(diag['selected']) == 0 and diag['delta'] == 0
