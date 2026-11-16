import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from solver import Solver  # noqa: E402

pytestmark = pytest.mark.skipif('S313_SOLVER_LIB' not in os.environ, reason='frozen solver library lives on Galvani')


def test_real_native_cpu_solver_rejects_missing_class():
    solver = Solver(os.environ['S313_SOLVER_LIB'])
    with pytest.raises(ValueError, match='three classes'):
        solver.fit(np.ones((2, 3)), np.array([0, 1]))
