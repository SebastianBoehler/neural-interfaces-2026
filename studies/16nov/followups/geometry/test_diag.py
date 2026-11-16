import unittest

import numpy as np
import torch

from diag import logits_delta, stream
from spdlog import spd_log


class DiagonalGeometryTests(unittest.TestCase):
    def test_zero_and_scalar_match_exact_matrix_log(self):
        torch.manual_seed(7)
        x = torch.randn(4, 2, 3, 3, dtype=torch.float64)
        matrices = x @ x.transpose(-1, -2) + .2 * torch.eye(3, dtype=torch.float64)
        head = torch.randn(3, 2, 6, dtype=torch.float64)
        zero = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        base = spd_log(matrices).detach()
        delta = logits_delta(matrices, base, head, zero, scalar=False)
        torch.testing.assert_close(delta, torch.zeros_like(delta), atol=1e-12, rtol=0)
        amount = torch.tensor([.2], dtype=torch.float64, requires_grad=True)
        exact = logits_delta(matrices, base, head, amount.expand(3), scalar=False)
        fast = logits_delta(matrices, base, head, amount, scalar=True)
        torch.testing.assert_close(exact, fast, atol=1e-10, rtol=1e-10)

    def test_causal_prefix_and_reset_without_labels(self):
        rng = np.random.default_rng(5)
        x = torch.tensor(rng.normal(size=(41, 2, 3, 3)), dtype=torch.float64)
        matrices = x @ x.transpose(-1, -2) + .4 * torch.eye(3, dtype=torch.float64)
        p = rng.dirichlet([2, 3, 4], size=41)
        head = torch.tensor(rng.normal(size=(3, 2, 6)), dtype=torch.float64)
        zero, _ = stream(matrices, p, head, 'zero', 0.)
        self.assertTrue(np.array_equal(zero, np.zeros_like(zero)))
        full, status = stream(matrices, p, head, 'diag', .5)
        short, _ = stream(matrices[:36], p[:36], head, 'diag', .5)
        repeat, _ = stream(matrices, p, head, 'diag', .5)
        np.testing.assert_allclose(full[:36], short, atol=1e-12)
        self.assertTrue(np.array_equal(full, repeat))
        self.assertTrue(np.array_equal(full[:32], np.zeros_like(full[:32])))
        self.assertGreater(status['grad_max'], 0)
        self.assertLessEqual(status['bias_norm'], .500001)


if __name__ == '__main__':
    unittest.main()
