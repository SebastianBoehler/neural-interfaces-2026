"""Compare the independent native analytical and autograd implementations."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel import Kernel  # noqa: E402

pytestmark = pytest.mark.skipif('S311_LIB' not in os.environ, reason='native library is built on Galvani')


def test_objective_gradient_match_and_invalid_input_fails():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(12, 7))
    y = np.tile(np.arange(3), 4).astype(np.int64)
    theta = rng.normal(size=(3, 8))
    lib = Kernel(os.environ['S311_LIB'])
    loss_a, grad_a = lib.objective(x, y, theta, analytical=False)
    loss_b, grad_b = lib.objective(x, y, theta, analytical=True)
    assert abs(loss_a - loss_b) <= 1e-9
    assert np.max(np.abs(grad_a - grad_b)) <= 1e-9
    x[0, 0] = np.nan
    with pytest.raises(RuntimeError, match='finite'):
        lib.objective(x, y, theta, analytical=True)
