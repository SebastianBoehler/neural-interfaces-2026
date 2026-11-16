import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spectral import cross_features, extract  # noqa: E402


def test_phase_lag_changes_only_signed_imaginary_cross_view():
    f = np.zeros((1, 41, 6), dtype=np.complex128)
    f[:, 0, :] = 1
    f[:, 1, :] = 1j
    power, real, imag = cross_features(f, 1.0)
    opposite = f.copy()
    opposite[:, 1, :] *= -1
    p2, r2, i2 = cross_features(opposite, 1.0)
    assert np.array_equal(power, p2)
    assert np.array_equal(real, r2)
    assert np.array_equal(imag, -i2)
    assert np.max(np.abs(imag)) > .9
    common_phase = cross_features(f * np.exp(.7j), 1.0)
    for a, b in zip((power, real, imag), common_phase):
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)


def test_zero_lag_and_common_average_reference():
    f = np.zeros((1, 41, 6), dtype=np.complex128)
    f[:, :2, :] = 1
    assert not np.any(cross_features(f, 1.0)[2])
    t = np.arange(480) / 120
    x = np.broadcast_to(np.sin(2*np.pi*10*t), (1, 41, 480)).copy()
    power, real, imag = extract(x)
    assert power.shape == (1, 861) and real.shape == imag.shape == (1, 17220)
    assert np.isfinite(power).all()
    assert np.max(np.abs(real)) < 1e-15 and np.max(np.abs(imag)) < 1e-15


def test_window_extraction_is_stateless_and_rejects_nonfinite():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(3, 41, 480))
    original = extract(x)
    reordered = extract(x[::-1])
    for a, b in zip(original, reordered):
        np.testing.assert_allclose(a, b[::-1], atol=0, rtol=0)
    x[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        extract(x)
