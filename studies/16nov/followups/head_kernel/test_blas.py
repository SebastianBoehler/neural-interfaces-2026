"""LP64 OpenBLAS objective and gradient must match the original native API."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blas import Blas  # noqa: E402
from kernel import Kernel  # noqa: E402

pytestmark = pytest.mark.skipif('S311_BLAS_LIB' not in os.environ, reason='BLAS library is built on Galvani')


def test_openblas_gradient_matches_native_on_nonzero_parameters():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(12, 7))
    y = np.tile(np.arange(3), 4).astype(np.int64)
    theta = rng.normal(size=(3, 8))
    ref, ref_grad = Kernel(os.environ['S311_LIB']).objective(x, y, theta, analytical=False)
    actual, gradient = Blas(os.environ['S311_BLAS_LIB']).objective(x, y, theta)
    assert abs(ref - actual) <= 1e-9
    assert np.max(np.abs(ref_grad - gradient)) <= 1e-9
