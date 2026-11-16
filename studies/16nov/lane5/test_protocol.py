import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'experiments'))
import protocol
from run import ORDERS, ROOT, orders, stream, validate_fit_status
from score import without_labels
from parity import session_checks


class FakeSolver:
    def __init__(self):
        self.windows = 0
        self.updates = 0
        self.last_prefix = 0

    def clone(self):
        clone = FakeSolver()
        clone.windows, clone.updates, clone.last_prefix = self.windows, self.updates, self.last_prefix
        return clone

    def reset(self):
        self.windows = self.updates = self.last_prefix = 0

    def step(self, x, update):
        if update and self.windows >= 40 and self.windows % 20 == 0:
            self.updates += 1
            self.last_prefix = min(64, self.windows)
        self.windows += 1
        return np.array([float(x[0, 0]), .01 * self.updates, 0], np.float32)

    def status(self):
        return {'windows': self.windows, 'updates': self.updates, 'prefix_count': self.last_prefix,
                'max_grad': .1 if self.updates else 0., 'delta': .01 if self.updates else 0.}

    def predict(self, x):
        return np.stack([np.array([float(row[0, 0]), 0, 0], np.float32) for row in x])

    def close(self):
        pass


class BrokenDisabledSolver(FakeSolver):
    def clone(self):
        clone = BrokenDisabledSolver()
        clone.windows, clone.updates, clone.last_prefix = self.windows, self.updates, self.last_prefix
        return clone

    def step(self, x, update):
        return super().step(x, True)


class ProtocolTests(unittest.TestCase):
    def test_consistency_gradient_must_be_active_only_in_treatment(self):
        status = {'steps': 1, 'consistency_sum': .2, 'projection_grad_max': .1, 'consistency_grad_max': .01}
        validate_fit_status(status, 3, 1)
        with self.assertRaises(ValueError):
            validate_fit_status({**status, 'consistency_grad_max': 0.}, 3, 1)
        with self.assertRaises(ValueError):
            validate_fit_status({**status, 'consistency_sum': 0.}, 3, 0)
        validate_fit_status({**status, 'consistency_sum': 0., 'consistency_grad_max': 0.}, 3, 0)

    def test_real_person_folds_and_eight_order_indices(self):
        d = without_labels(ROOT)
        folds = list(protocol.folds(d, 'all_loso'))
        self.assertEqual(len(folds), 10)
        pinned = orders(ROOT, len(d['subject']))
        self.assertEqual(tuple(pinned), ORDERS)
        for held, tr, te in folds:
            self.assertEqual(set(d['subject'][te]), {'Dreyer2026Proteus/' + held})
            self.assertEqual(len(set(d['subject'][tr])), 9)
            for perm in pinned.values():
                self.assertTrue(np.array_equal(np.sort(perm[te[perm]]), np.flatnonzero(te)))

    def test_update_disabled_identity_and_first_update_position(self):
        a, b, status = stream(FakeSolver(), np.ones((61, 41, 480), np.float32))
        self.assertTrue(np.array_equal(a[:40], b[:40]))
        self.assertFalse(np.array_equal(a[40], b[40]))
        self.assertEqual(status['adaptive']['updates'], 2)
        session_checks(FakeSolver(), np.ones((61, 41, 480), np.float32))
        with self.assertRaises(ValueError):
            session_checks(BrokenDisabledSolver(), np.ones((61, 41, 480), np.float32))


if __name__ == '__main__':
    unittest.main()
