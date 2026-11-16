import copy
import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from bridge import Session, raw_array, labels_array


class FakeSolver:
    def __init__(self):
        self.windows = []

    def clone(self):
        result = FakeSolver()
        result.windows = list(self.windows)
        return result

    def reset(self):
        self.windows.clear()

    def step(self, x, update):
        value = float(x[0, 0])
        result = np.array([sum(self.windows), value, int(update)], dtype=np.float32)
        self.windows.append(value)
        return result


class BridgeTests(unittest.TestCase):
    def test_entry_points_import_under_isolated_python(self):
        root = Path(__file__).resolve().parents[3]
        for name in ('run.py', 'parity.py', 'smoke.py'):
            entry = Path(__file__).with_name(name)
            result = subprocess.run([sys.executable, '-I', '-c', 'import runpy,sys; runpy.run_path(sys.argv[1], run_name="check")', str(entry)],
                                    env=dict(os.environ, NI26_WS=str(root)), capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_raw_layout_and_finite_values_are_required(self):
        x = raw_array(np.zeros((2, 41, 480), dtype=np.float64))
        self.assertEqual(x.dtype, np.float32)
        self.assertTrue(x.flags.c_contiguous)
        with self.assertRaises(ValueError):
            raw_array(np.zeros((2, 40, 480)))
        bad = np.zeros((1, 41, 480)); bad[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            raw_array(bad)
        with self.assertRaises(ValueError):
            labels_array([0, 3], 2)

    def test_sessions_clone_reset_and_only_receive_current_window(self):
        population = FakeSolver()
        left, right = Session(population), Session(population)
        x = np.ones((41, 480), dtype=np.float32)
        self.assertEqual(left.step(x, True).tolist(), [0., 1., 1.])
        branch = copy.deepcopy(left)
        self.assertEqual(left.step(2 * x, True).tolist(), [1., 2., 1.])
        self.assertEqual(right.step(3 * x, False).tolist(), [0., 3., 0.])
        self.assertEqual(branch.step(4 * x, True).tolist(), [1., 4., 1.])
        left.reset()
        self.assertEqual(left.step(x, True).tolist(), [0., 1., 1.])
        self.assertEqual(population.windows, [])


if __name__ == '__main__':
    unittest.main()
