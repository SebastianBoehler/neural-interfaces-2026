"""Pinned source-only dependencies and seven-source people for325."""
import json
import sys
from pathlib import Path
import numpy as np

PREVIOUS = Path('/mnt/lustre/work/gehler/gfh527/ni26_visual324_v2_20261117')
sys.path.insert(0, str(PREVIOUS))
from shared import build, check, digest, features, labels, load_data, offline, sha, state_sha
from calibrate import load
from display import Display

PINS = {'source.sha256': '84c7004a5d2202ac1d4d4c490f146b9058863edda38a49db30223596ef5a281e',
        'renderer.npz': 'a81748fe32a825f727f99db9a4036d773cea7861690a0e546aa5222cbd8972b2',
        'source_features.npz': '744587e6323d3587ae9c521e982279219c90dd3f277526e2122583948cda2a0b',
        'source_results.json': 'f17c99d5938f284ae3cba04b3a6dfb048ef8d67e04be7a28398284271a952293'}


def setup(root):
    check(root)
    for name, expected in PINS.items():
        if sha(PREVIOUS / name) != expected: raise ValueError('324 dependency changed:' + name)
    for line in (PREVIOUS / 'source.sha256').read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        if sha(PREVIOUS / name) != expected: raise ValueError('324 source changed')
    d, fit, val, people = load_data()
    renderer, metadata = load(PREVIOUS, fit)
    return d, fit, val, people, renderer, metadata


def fixed_features(rows, variant):
    with np.load(PREVIOUS / 'source_features.npz') as z:
        if not np.array_equal(z['index'], rows): raise ValueError('fixed feature rows differ')
        return z['raw287'], z[f'{variant}_41_intensity']


def reference():
    return json.loads((PREVIOUS / 'source_results.json').read_text())['strong_reference']
