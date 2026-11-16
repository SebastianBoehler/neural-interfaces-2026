import importlib.util
import unittest
from pathlib import Path
import numpy as np

PATH = Path(__file__).with_name('mask.py')

class MaskTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('mask', PATH)
        self.m = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.m)

    def test_binary_threshold_and_exact_unchanged_rows(self):
        base = np.array([[.5,.3,.2],[.81,.1,.09],[.6,.3,.1]])
        added = np.array([[.4,.4,.2],[.3,.4,.3],[.3,.2,.5]])
        out, stats = self.m.apply(base, added, .6)
        np.testing.assert_array_equal(out, np.vstack([added[0],base[1],added[2]]))
        self.assertEqual(stats['eligible'], 2); self.assertEqual(stats['changed'], 2)
    def test_registered_zero_keeps_all_rows_and_unregistered_noop_fails(self):
        base = np.array([[.5,.3,.2],[.81,.1,.09]])
        out, stats = self.m.apply(base, base.copy(), .6, zero=True)
        np.testing.assert_array_equal(out, base)
        self.assertEqual(stats['eligible'], 1)
        self.assertEqual(stats['changed'], 0)
        with self.assertRaises(ValueError): self.m.apply(base, base.copy(), .6)
        with self.assertRaises(ValueError): self.m.apply(base, base + .01, .6, zero=True)

    def test_invalid_probabilities_and_silent_noops_fail(self):
        for base, added in [(np.array([[.7,.4,-.1]]), np.array([[.3,.3,.4]])),
                            (np.array([[.9,.05,.05]]), np.array([[.3,.3,.4]])),
                            (np.array([[.5,.3,.2]]), np.array([[.5,.3,.2]]))]:
            with self.assertRaises(ValueError): self.m.apply(base, added, .6)

if __name__ == '__main__': unittest.main()
