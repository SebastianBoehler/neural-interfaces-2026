import unittest

import numpy as np
import torch

from adapt import stream
from slopes import coefficients


class GeometryTests(unittest.TestCase):
    def test_logit_and_pc_directions_match_full_feature_shift(self):
        rng = np.random.default_rng(3)
        channels, bands, blocks = 3, 2, 3
        width = channels * (channels + 1) // 2
        dim = bands * blocks * width + 4
        head_weight = rng.normal(size=(3, dim)); head_scale = rng.uniform(.5, 2, size=dim)
        pc = rng.normal(size=(5, dim)); pc_scale = rng.uniform(.5, 2, size=dim)
        core, view = coefficients(head_weight, head_scale, pc, pc_scale, channels, bands, blocks)
        a = np.array([.2, -.3]); x = rng.normal(size=dim); changed = x.copy()
        ii, jj = np.triu_indices(channels); diag = np.flatnonzero(ii == jj)
        for band in range(bands):
            for block in range(blocks):
                changed[(band * blocks + block) * width + diag] += a[band]
        np.testing.assert_allclose(core @ a, head_weight @ ((changed - x) / head_scale), atol=1e-12)
        np.testing.assert_allclose(view @ a, pc @ ((changed - x) / pc_scale), atol=1e-12)

    def test_zero_prefix_reset_and_active_updates(self):
        rng = np.random.default_rng(1)
        p = rng.dirichlet([2, 3, 4], 72)
        core = rng.normal(scale=.2, size=(3, 17))
        view = rng.normal(scale=.1, size=(128, 17))
        zero, zpc, _ = stream(p, core, view, 'zero')
        self.assertTrue(np.array_equal(zero, p))
        self.assertTrue(np.array_equal(zpc, np.zeros_like(zpc)))
        updated, pcs, status = stream(p, core, view, 'band')
        prefix, pc_prefix, _ = stream(p[:40], core, view, 'band')
        repeat, _, _ = stream(p, core, view, 'band')
        np.testing.assert_allclose(updated[:40], prefix, atol=1e-12)
        np.testing.assert_allclose(pcs[:40], pc_prefix, atol=1e-12)
        self.assertTrue(np.array_equal(updated, repeat))
        self.assertTrue(np.array_equal(updated[:32], p[:32]))
        self.assertGreater(np.max(np.abs(updated[32:] - p[32:])), 1e-8)
        self.assertGreater(status['grad_max'], 0)
        self.assertLessEqual(status['bias_norm'], .500001)


if __name__ == '__main__':
    unittest.main()
