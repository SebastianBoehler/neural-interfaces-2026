import unittest
import numpy as np
import rank_audit as r


class RankAuditTest(unittest.TestCase):
    def test_car_null_and_valid_scale(self):
        p = np.eye(41) - np.ones((41, 41)) / 41
        sample = np.diag(np.linspace(1, 2, 41))
        summary = r.record(sample, p)
        self.assertEqual(summary['raw']['below']['1e-06'], 0)
        self.assertEqual(summary['car']['below']['1e-06'], 1)
        with self.assertRaises(ValueError): r.spectra(np.zeros((41, 41)))

    def test_centered_covariance_uses_only_window(self):
        x = np.random.default_rng(2).normal(size=(2, 41, 480))
        a = r.window_covariances(x)
        np.testing.assert_allclose(a[0], np.cov(x[0], bias=True), rtol=1e-12, atol=1e-12)
        x[1] *= 4
        np.testing.assert_array_equal(a[0], r.window_covariances(x)[0])


if __name__ == '__main__': unittest.main()
