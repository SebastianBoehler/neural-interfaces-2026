"""Pre-registered representation ensembles and matched current-system contrasts."""
import hashlib
from pathlib import Path

ORDERS = ('within0', 'within1', 'within2', 'within3', 'full0', 'full1', 'runs0', 'runs1')
BASES = ('B2_pool', 'B2_s345', 'B2_s012', 'B2_s0', 'B2_s1', 'B2_s2')
GROUPS = {
    '317': {'families': ('auxa317','auxb317','auxc317'), 'arms': ('A','B','C'),
            'contrasts': [('C','B'),('C','A'),('B','A')]},
}


def recipes(group):
    cfg = GROUPS[group]
    return {'control': ('none', 'none', False),
            'zero_repr': (cfg['families'][0], 'none', True),
            **{a: (f, 'none', False) for a, f in zip(cfg['arms'], cfg['families'])}}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as src:
        for b in iter(lambda: src.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()
