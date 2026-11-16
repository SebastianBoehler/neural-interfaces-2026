"""Portable, source-fitted inference artifacts for the C++ LibTorch bridge."""

import copy

import numpy as np
import torch
from torch import nn


class StructuredInference(nn.Module):
    def __init__(self, model, mean, scale):
        super().__init__()
        self.model = copy.deepcopy(model).cpu().eval()
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float64))
        self.register_buffer('scale', torch.as_tensor(scale, dtype=torch.float64))

    def forward(self, raw):
        x = ((raw - self.mean) / self.scale).float()
        return self.model(x).softmax(dim=1)


class RidgeInference(nn.Module):
    def __init__(self, fitted):
        super().__init__()
        mapper = fitted['mapper']
        self.arm = mapper.arm
        for name, value in [('mean', fitted['mean']), ('scale', fitted['scale']),
                            ('projection', mapper.projection), ('bias', mapper.bias),
                            ('input_mean', mapper.input_mean), ('input_scale', mapper.input_scale),
                            ('map_mean', mapper.map_mean), ('map_scale', mapper.map_scale),
                            ('weights', fitted['weights'])]:
            tensor = torch.as_tensor(value, dtype=torch.float32 if name == 'projection' else torch.float64)
            self.register_buffer(name, tensor)
        self.temperature = float(fitted['temperature'])

    def forward(self, raw):
        x = ((raw - self.mean) / self.scale).float()
        u = (torch.matmul(x, self.projection).double() - self.input_mean) / self.input_scale
        if self.arm == 'linear':
            z = u
        elif self.arm == 'quadratic':
            z = torch.cat((u, .5 * (u * u - 1)), dim=1)
        else:
            z = np.sqrt(2) * torch.cos(u + self.bias)
        z = (z - self.map_mean) / self.map_scale
        logits = (z @ self.weights[:-1] + self.weights[-1]) / self.temperature
        return logits.softmax(dim=1)


def _save(module, example, target):
    example = np.asarray(example, dtype=np.float64)
    if example.ndim != 2 or not len(example) or not np.isfinite(example).all():
        raise ValueError('export requires finite real feature examples')
    module = torch.jit.trace(module.eval(), torch.from_numpy(example))
    with open(target, 'xb') as output:
        torch.jit.save(module, output)


def structured_export(model, mean, scale, example, target):
    _save(StructuredInference(model, mean, scale), example, target)


def ridge_export(fitted, example, target):
    _save(RidgeInference(fitted), example, target)
