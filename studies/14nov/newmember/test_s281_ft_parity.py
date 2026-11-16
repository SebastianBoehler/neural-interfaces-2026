"""Contract tests for the one-window-21 implementation diagnostic."""
import importlib.util
import unittest
from pathlib import Path
import torch
from torch import nn

spec = importlib.util.spec_from_file_location('s281_ft_parity', Path(__file__).with_name('s281_ft_parity.py'))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)


class Models(nn.Module):
    def __init__(self, traced=False):
        super().__init__(); self.bn = nn.Module(); self.layer = nn.Linear(2, 2)
        if traced:
            self.bn.register_buffer('w', torch.ones(2)); self.bn.register_buffer('b', torch.zeros(2))
        else:
            self.bn.register_parameter('weight', nn.Parameter(torch.ones(2)))
            self.bn.register_parameter('bias', nn.Parameter(torch.zeros(2)))
        self.layer.bias.requires_grad_(False)


class ParityChecks(unittest.TestCase):
    def test_maps_affine_and_requires_all_parameters(self):
        eager, trace = Models(), Models(True)
        result = mod.parameter_map(eager, trace)
        self.assertEqual(set(result), {'bn.w', 'bn.b', 'layer.weight'})
        trace.bn.register_buffer('extra', torch.zeros(2))
        trace.layer.weight.requires_grad_(False)
        with self.assertRaisesRegex(AssertionError, 'trainable/frozen'):
            mod.parameter_map(eager, trace)

    def test_gradients_require_matching_names_and_nonzero_finite_values(self):
        p, q = nn.Parameter(torch.ones(2)), nn.Parameter(torch.ones(2)); p.grad = torch.tensor([1., 2.])
        mapping = {'x': (p, q)}
        self.assertEqual(mod.grad_rel(mapping, {'x': p.grad.clone()}), 0.)
        with self.assertRaisesRegex(AssertionError, 'mapping'):
            mod.grad_rel(mapping, {})
        with self.assertRaisesRegex(AssertionError, 'zero gradient'):
            mod.grad_rel(mapping, {'x': torch.zeros(2)})
        with self.assertRaisesRegex(AssertionError, 'nonfinite'):
            mod.grad_rel(mapping, {'x': torch.tensor([float('nan'), 0.])})

    def test_finite_difference_rejects_nan(self):
        with self.assertRaisesRegex(AssertionError, 'nonfinite'):
            mod.finite_max(torch.tensor([float('nan')]), torch.zeros(1))


if __name__ == '__main__': unittest.main()
