"""Fixed §313 source-view compositions; no session or person state."""

import numpy as np

ARMS = ('BASE', 'POWER', 'REAL', 'IMAG', 'BOTH', 'SPECTRAL')
ORDERS = ('within0', 'within1', 'within2', 'within3', 'full0', 'full1', 'runs0', 'runs1')
VIEWS = {'strong': 44239, 'power': 861, 'real': 17220, 'imag': 17220}
PARTS = {'BASE': ('strong',), 'POWER': ('strong', 'power'), 'REAL': ('strong', 'power', 'real'),
         'IMAG': ('strong', 'power', 'imag'), 'BOTH': ('strong', 'power', 'real', 'imag'),
         'SPECTRAL': ('power', 'real', 'imag')}


def compose(strong, power, real, imag, arm):
    if arm not in ARMS:
        raise ValueError('unknown registered arm')
    views = dict(strong=strong, power=power, real=real, imag=imag)
    parts, n = [], None
    for name in PARTS[arm]:
        value = np.asarray(views[name])
        if (value.ndim != 2 or value.shape[1] != VIEWS[name] or not len(value) or
                (n is not None and len(value) != n) or not np.isfinite(value).all()):
            raise ValueError(f'invalid {name} view')
        n = len(value)
        parts.append(value)
    return np.ascontiguousarray(np.concatenate(parts, axis=1), dtype=np.float64)
