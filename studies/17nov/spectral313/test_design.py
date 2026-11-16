import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from design import ARMS, compose  # noqa: E402


def test_each_view_is_applied_without_changing_strong_features():
    strong = np.full((2, 44239), 7.0)
    power = np.full((2, 861), 11.0)
    real = np.full((2, 17220), 13.0)
    imag = np.full((2, 17220), 17.0)
    sizes = {'BASE': 44239, 'POWER': 45100, 'REAL': 62320,
             'IMAG': 62320, 'BOTH': 79540, 'SPECTRAL': 35301}
    assert set(ARMS) == set(sizes)
    for arm, size in sizes.items():
        result = compose(strong, power, real, imag, arm)
        assert result.shape == (2, size)
        if arm != 'SPECTRAL':
            assert np.array_equal(result[:, :44239], strong)
    assert np.all(compose(strong, power, real, imag, 'REAL')[:, -17220:] == 13)
    assert np.all(compose(strong, power, real, imag, 'IMAG')[:, -17220:] == 17)
    assert np.all(compose(strong, power, real, imag, 'SPECTRAL')[:, :861] == 11)


def test_composition_rejects_missing_or_invalid_views():
    with pytest.raises(ValueError, match='view'):
        compose(np.zeros((1, 44239)), np.zeros((1, 861)), np.zeros((1, 1)),
                np.zeros((1, 17220)), 'REAL')
    with pytest.raises(ValueError, match='arm'):
        compose(None, None, None, None, 'unknown')
