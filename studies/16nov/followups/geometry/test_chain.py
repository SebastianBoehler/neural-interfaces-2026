import unittest
from pathlib import Path

import numpy as np

from chain import patch_chain, validate_delta


class ChainTests(unittest.TestCase):
    def test_injection_requires_exactly_one_anchor(self):
        original = 'def chain():\n    HB = {"B2_pool": 1}\n    return HB\n'
        patched = patch_chain(original)
        self.assertIn('if arm == "V6Fr"', patched)
        self.assertIn('GEO_DELTA', patched)
        compile(patched, '<test>', 'exec')
        with self.assertRaises(ValueError):
            patch_chain(original + original)
        root = Path(__file__).resolve().parents[4]
        source = (root / 'studies/13nov/shufcombo.py').read_text()
        compile(patch_chain(source), '<actual-chain>', 'exec')

    def test_delta_guard_accepts_true_zero_and_rejects_bad_sum(self):
        a = np.zeros((5, 3), np.float64)
        validate_delta(a, 5, zero=True)
        with self.assertRaises(ValueError):
            validate_delta(np.ones((5, 3)), 5, zero=False)
        with self.assertRaises(ValueError):
            validate_delta(a, 5, zero=False)


if __name__ == '__main__':
    unittest.main()
