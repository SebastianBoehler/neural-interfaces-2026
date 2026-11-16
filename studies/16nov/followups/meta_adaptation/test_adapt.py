"""Exercise exact gradient, bounded updates, zero identity and session isolation."""
import unittest
import torch
from adapt import FEATURES, MUTABLE, update, Session


class AdaptationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.anchor = torch.randn(3, FEATURES, dtype=torch.float64) * .01
        self.bias = torch.randn(3, dtype=torch.float64) * .1
        self.x = torch.randn(64, FEATURES, dtype=torch.float64)

    def test_meta_gradient_and_frozen_coordinates(self):
        parameter = torch.tensor(.5493061443340549, dtype=torch.float64, requires_grad=True)
        head, stats = update(self.anchor, self.anchor.clone(), self.bias, self.x[:8], parameter.tanh(), create_graph=True)
        self.assertGreater(stats['gradient_max'], 0.)
        self.assertLessEqual(stats['delta_ratio_max'], .03000001)
        self.assertTrue(torch.equal(head[:, MUTABLE:], self.anchor[:, MUTABLE:]))
        query = torch.randn(4, FEATURES, dtype=torch.float64)
        outer = torch.nn.functional.cross_entropy(query @ head.T + self.bias, torch.tensor([0,1,2,0]))
        gradient = torch.autograd.grad(outer, parameter)[0]
        self.assertTrue(torch.isfinite(gradient).item())
        self.assertGreater(abs(float(gradient)), 1e-12)

    def test_zero_no_future_and_reset(self):
        zero = Session(self.anchor, self.bias, torch.tensor(0., dtype=torch.float64))
        baseline = Session(self.anchor, self.bias, torch.tensor(1., dtype=torch.float64), enabled=False)
        for row in self.x[:43]:
            self.assertTrue(torch.equal(zero.step(row), baseline.step(row)))
        self.assertEqual(zero.updates, 1)
        self.assertTrue(torch.equal(zero.head, self.anchor))
        session = Session(self.anchor, self.bias, torch.tensor(.5, dtype=torch.float64))
        before = [session.step(row) for row in self.x[:43]]
        session.reset()
        again = [session.step(row) for row in self.x[:43]]
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,again)))
        self.assertEqual(session.updates, 1)


if __name__ == '__main__': unittest.main()
