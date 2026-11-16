"""Non-fitting contract tests for the registered stage diagnostic."""
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np

spec = importlib.util.spec_from_file_location('s281_fold_stage_diag', Path(__file__).with_name('s281_fold_stage_diag.py'))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)


class StageDiagTest(unittest.TestCase):
    def test_head_recovery_matches_direct_causal_formula(self):
        rng = np.random.default_rng(0); U = rng.normal(size=(8, 256)); W = rng.normal(size=(3, 256)); c = rng.normal(size=3)
        head = SimpleNamespace(coef_=W, intercept_=c); got = mod.head_route(head, U, np.ones(8))
        expected = []
        for t in range(8):
            A = U[:t+1]; mean = A.mean(0); v = (U[t]-mean)/np.maximum(A.std(0),1e-3) if t >= 4 else U[t]-mean
            expected.append(mod.softmax(v[None] @ W.T + c)[0])
        self.assertLessEqual(mod.diff(got, expected, 'head'), 1e-8)
        with self.assertRaises(AssertionError): mod.head_route(head, U, np.ones(7))

    def test_nonfinite_and_wrong_shape_fail(self):
        with self.assertRaisesRegex(AssertionError, 'nonfinite'): mod.diff([np.nan], [0.], 'x')
        with self.assertRaisesRegex(AssertionError, 'shape'): mod.diff([1.,2.], [1.], 'x')
        with self.assertRaisesRegex(AssertionError, '> 1e-8'): mod.diff([1.], [1.001], 'x')


if __name__ == '__main__': unittest.main()
