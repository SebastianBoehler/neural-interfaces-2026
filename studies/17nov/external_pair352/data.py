"""Immutable source channels, qualified target groups, and delivery orders."""
import csv
import gzip
import hashlib

import numpy as np

CHANNELS = 'F3 Fz F4 FC3 FCz FC4 C3 Cz C4 CP3 CPz CP4 P5 P3 P1 Pz P2 P4 P6'.split()
ORDERS = tuple((s, k) for s, ns in (('within', 4), ('full', 2), ('runs', 2)) for k in range(ns))
SCHEME_ID = {'full': 0, 'runs': 1, 'within': 2}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def load_source(root):
    path = root / 'data/proteus_windows.npz'
    with np.load(path) as z:
        d = {k: z[k] for k in ('X', 'y', 'subject', 'session', 'interface', 'run', 'task', 'ch_names')}
    assert len(set(d['subject'])) == 10
    d['X'] = d['X'][:, [list(d['ch_names']).index(c) for c in CHANNELS]].copy()
    d['stream'] = np.char.add(np.char.add(d['subject'], '|'), d['session'])
    d['group'] = np.char.add(np.char.add(d['task'], '|'), d['run'].astype(str))
    return d


def load_target(root):
    path = root / 'data/scherer2015/windows.npz'
    manifest = root / 'runs/diag/external_semantics351/scherer_rows.tsv.gz'
    with gzip.open(manifest, 'rt') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    with np.load(path) as z:
        names = z['ch_names'].tolist()
        native = set(names) - set(z['interpolated'])
        assert set(CHANNELS) == native
        d = {k: z[k] for k in ('subject', 'session')}
        d['X'] = z['X'][:, [names.index(c) for c in CHANNELS]].copy()
    assert len(rows) == len(d['X']) == 2130
    assert [int(r['row']) for r in rows] == list(range(2130))
    for i, r in enumerate(rows):
        assert d['subject'][i] == f"Scherer2015/{r['person']}" and d['session'][i] == r['session']
    d['stream'] = np.char.add(np.char.add(d['subject'], '|'), d['session'])
    d['group'] = np.array([r['run'] for r in rows])
    return d


def permutation(stream, group, scheme, seed):
    rng = np.random.default_rng([seed, 0, SCHEME_ID[scheme]])
    p = np.arange(len(stream))
    for session in dict.fromkeys(stream):
        rows = np.flatnonzero(stream == session)
        if scheme == 'full':
            p[rows] = rows[rng.permutation(len(rows))]
        else:
            blocks = [rows[group[rows] == g] for g in dict.fromkeys(group[rows])]
            if scheme == 'runs':
                blocks = [blocks[i] for i in rng.permutation(len(blocks))]
            else:
                blocks = [b[rng.permutation(len(b))] for b in blocks]
            p[rows] = np.concatenate(blocks)
    assert np.array_equal(np.sort(p), np.arange(len(stream)))
    return p


def source_orders(root, d):
    out = {}
    with np.load(root / 'runs/shuffle_perms.npz') as z:
        for scheme, seed in ORDERS:
            p = z[f'{scheme}|{seed}|all_loso']
            expected = permutation(d['stream'], d['group'], scheme, seed)
            assert np.array_equal(p, expected), (scheme, seed, 'pinned order mismatch')
            out[f'{scheme}{seed}'] = p
    return out
