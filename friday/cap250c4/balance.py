"""Move represented pseudo-class CE coefficient masses symmetrically."""
import numpy as np


def class_mass(labels, weights):
    return np.bincount(labels, weights=weights, minlength=3)


def adjust(arm, labels, weights):
    mass = class_mass(labels, weights)
    present = mass > 0
    if arm == 'stock' or present.sum() < 2:
        return weights
    power = {'primary': -.5, 'placebo': .5}[arm]
    factor = np.ones(3)
    factor[present] = np.clip((mass[present]/mass[present].mean())**power, .5, 2.)
    adjusted = weights*factor[labels]
    return adjusted*(weights.sum()/adjusted.sum())
