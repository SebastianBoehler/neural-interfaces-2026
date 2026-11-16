"""Registered strong-feature head definitions; fitting uses source rows only."""

import numpy as np
import torch
from torch import nn


class Structured(nn.Module):
    def __init__(self, nonlinear):
        super().__init__()
        self.block = nn.Linear(861, 16)
        self.slow = nn.Linear(328, 16)
        self.classifier = nn.Linear(17 * 2 * 16 + 16, 3)
        self.nonlinear = bool(nonlinear)

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != 44239:
            raise ValueError('expected N × 44239 strong features')
        blocks = self.block(x[:, :43911].reshape(-1, 17, 3, 861))
        slow = self.slow(x[:, 43911:])
        if self.nonlinear:
            blocks = torch.tanh(blocks)
            slow = torch.tanh(slow)
        bands = torch.cat((blocks[:, :, 0], blocks[:, :, 1] - blocks[:, :, 2]), dim=2)
        return self.classifier(torch.cat((bands.flatten(1), slow), dim=1))


def fit_scale(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('expected a nonempty finite training matrix')
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return mean, scale


def ridge_fit(x, y, strength):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if x.ndim != 2 or y.shape != (len(x),) or not len(x) or not np.isfinite(x).all():
        raise ValueError('invalid ridge inputs')
    if not np.isin(y, (0, 1, 2)).all() or strength <= 0:
        raise ValueError('invalid labels or ridge strength')
    offset = x.mean(axis=0)
    target = np.eye(3)[y]
    average = target.mean(axis=0)
    centered = x - offset
    weights = np.linalg.solve(centered.T @ centered / len(x) + strength * np.eye(x.shape[1]),
                              centered.T @ (target - average) / len(x))
    return np.vstack((weights, average - offset @ weights))


def ridge_predict(x, weights, temperature):
    x = np.asarray(x, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if x.ndim != 2 or weights.shape != (x.shape[1] + 1, 3) or temperature <= 0:
        raise ValueError('invalid ridge prediction inputs')
    logits = (x @ weights[:-1] + weights[-1]) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    if not np.isfinite(probs).all():
        raise ValueError('nonfinite ridge predictions')
    return probs


class RandomMap:
    def __init__(self, arm, projection, bias, input_mean, input_scale, map_mean, map_scale):
        self.arm, self.projection, self.bias = arm, projection, bias
        self.input_mean, self.input_scale = input_mean, input_scale
        self.map_mean, self.map_scale = map_mean, map_scale

    def _map(self, x):
        u = (np.asarray(x) @ self.projection - self.input_mean) / self.input_scale
        if self.arm == 'linear':
            return u
        if self.arm == 'quadratic':
            return np.concatenate((u, .5 * (u * u - 1)), axis=1)
        if self.arm == 'fourier':
            return np.sqrt(2) * np.cos(u + self.bias)
        raise ValueError('unregistered random-feature arm')

    def transform(self, x):
        result = (self._map(x) - self.map_mean) / self.map_scale
        if not np.isfinite(result).all():
            raise ValueError('nonfinite random features')
        return result

    @classmethod
    def fit(cls, x, arm, seed, width=512):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or not len(x) or not x.shape[1] or width <= 0 or not np.isfinite(x).all():
            raise ValueError('expected nonempty finite random-map training matrix')
        rng = np.random.default_rng(seed)
        projection = rng.standard_normal((x.shape[1], width), dtype=np.float32) / np.sqrt(x.shape[1])
        bias = rng.uniform(0, 2 * np.pi, size=width)
        mean, scale = fit_scale(x @ projection)
        obj = cls(arm, projection, bias, mean, scale, 0, 1)
        obj.map_mean, obj.map_scale = fit_scale(obj._map(x))
        return obj
