import unittest

import numpy as np
import torch

from spdlog import spd_log


class SPDLogTests(unittest.TestCase):
    def test_repeated_eigenvalues_have_correct_finite_gradient(self):
        x = torch.eye(3, dtype=torch.float64, requires_grad=True)
        weight = torch.tensor([[1., 2., 3.], [2., -1., 4.], [3., 4., .5]], dtype=torch.float64)
        value = (spd_log(x) * weight).sum()
        grad, = torch.autograd.grad(value, x)
        self.assertTrue(torch.isfinite(grad).all())
        torch.testing.assert_close(grad, weight, rtol=1e-12, atol=1e-12)

    def test_gradient_matches_central_finite_difference(self):
        rng = np.random.default_rng(42)
        a = rng.normal(size=(4, 4)); x = torch.tensor(a @ a.T + .2 * np.eye(4), dtype=torch.float64, requires_grad=True)
        weight = torch.tensor(rng.normal(size=(4, 4)), dtype=torch.float64)
        direction = rng.normal(size=(4, 4)); direction = torch.tensor((direction + direction.T) / 2, dtype=torch.float64)
        value = (spd_log(x) * weight).sum(); grad, = torch.autograd.grad(value, x)
        h = 1e-6
        f = lambda z: float((spd_log(z) * weight).sum())
        finite = (f(x.detach() + h * direction) - f(x.detach() - h * direction)) / (2 * h)
        self.assertAlmostEqual(float((grad * direction).sum()), finite, delta=2e-7)


if __name__ == '__main__':
    unittest.main()
