"""Reuse exact passed319 assets/encoder and source-only label reader."""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

SHARED = Path('/mnt/lustre/work/gehler/gfh527/ni26_visual319_v1_20261117')
SOURCE_SHA = '0a80aeb046932421c78729ef2bcd2e9c30c9be78090f1f62c019d60789cdf943'
LABELS = Path('/mnt/lustre/work/gehler/gfh527/ni26_s306_expert_cpu_v2_20261116/source_labels.py')
LABEL_SHA = '35edab56fe754e6c77a81f54f2010baca7e0912af55af9564043b2a8bde8c4dc'
BASE = Path('/mnt/lustre/work/gehler/gfh527/ni26_raw312_20261117_matrix2')
BASE_SHA = '8275d60d30d618a1a90d184f504522682b252221e956356e5041bed2b3ac6559'
sys.path.insert(0, str(SHARED))
from assets import sha, verify
from encoder import build, features, offline, state_sha


def check(root):
    if sha(SHARED / 'source.sha256') != SOURCE_SHA or sha(BASE / 'source.sha256') != BASE_SHA or sha(LABELS) != LABEL_SHA:
        raise ValueError('shared source identity changed')
    for line in (SHARED / 'source.sha256').read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        if sha(SHARED / name) != expected:
            raise ValueError('shared319 code changed')
    if sha(root / 'assets/manifest.json') != 'fa57956e555af175cc10fb5c780aa20aaaaf57784c5a816272af32e001b7e3cc':
        raise ValueError('passed319 asset manifest changed')
    for line in (BASE / 'source.sha256').read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        if sha(BASE / name) != expected:
            raise ValueError('strong raw control dependency changed')
    return verify(root)


def labels(rows, n):
    if sha(LABELS) != LABEL_SHA:
        raise ValueError('selected-label reader changed')
    spec = importlib.util.spec_from_file_location('visual324_labels', LABELS)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.read_training_labels(DATA, rows, n)


DATA = Path('/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/data/proteus_windows.npz')
DATA_SHA = '2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327'


def load_data():
    import numpy as np
    if sha(DATA) != DATA_SHA:
        raise ValueError('official input tensor changed')
    with np.load(DATA, allow_pickle=False) as z:
        d = {k: z[k] for k in ('X', 'subject', 'session', 'ch_names')}
    people = np.array([str(s).split('/')[-1] for s in d['subject']])
    fit = np.flatnonzero(np.isin(people, [f'P{i:02d}' for i in range(4, 11)]))
    val = np.flatnonzero(np.isin(people, ['P02', 'P03']))
    if len(set(people[fit])) != 7 or set(people[val]) != {'P02', 'P03'} or np.any(people[np.r_[fit, val]] == 'P01'):
        raise ValueError('source7/two-validation/P01 exclusion differs')
    d['stream'] = np.char.add(np.char.add(d['subject'], '|'), d['session'])
    return d, fit, val, people


def digest(x):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
