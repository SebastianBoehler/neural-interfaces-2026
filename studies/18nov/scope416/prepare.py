"""Prepare label-blind deliveries and pin inputs from the two original studies."""
import csv
import gzip
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

base = Path('..')
repo = base / 'neural-interfaces-2026'
old = base / 'ni26_qualify385_v1_20261118'
full = base / 'ni26_cross411b_20261118'
for bank in (old, full):
    for manifest in ('sources.sha256', 'members.sha256'):
        subprocess.run(['sha256sum', '-c', '--status', manifest], cwd=bank, check=True)
assert (old/'members.sha256').read_bytes() == (full/'members.sha256').read_bytes()
for arm in ('old', 'cap'):
    shutil.copytree(old/arm, arm)
for name in ('members.sha256', 'inputs.sha256', 'data.py', 'worker.py', 'replay.py'):
    shutil.copyfile(old/name, name)
sys.path.insert(0, str(Path.cwd()))
from data import permutation

manifest = repo/'runs/diag/external_semantics351/scherer_rows.tsv.gz'
with gzip.open(manifest, 'rt') as f:
    records = list(csv.DictReader(f, delimiter='\t'))
with np.load(repo/'data/scherer2015/windows.npz') as z:
    people, days = z['subject'], z['session']
    assert len(set(z['interface'])) == 1 and (z['run'] == 0).all()
    assert np.array_equal(z['onset'], [float(r['cue_s']) for r in records])
assert np.array_equal(people, ['Scherer2015/'+r['person'] for r in records])
assert np.array_equal(days, [r['session'] for r in records])
assert [int(r['row']) for r in records] == list(range(2130))
streams = np.char.add(np.char.add(people, '|'), days)
groups = np.array([int(r['run']) for r in records])
orders = {}
for seed in (411, 1411):
    perm = permutation(streams, None, 'full', seed)
    orders[f'full{seed}'] = perm
    fixed, moved = np.empty_like(perm), np.empty_like(perm)
    for index, stream in enumerate(sorted(set(streams))):
        rows = np.flatnonzero(streams == stream)
        src = perm[rows]
        runs = sorted(set(groups[rows]))
        blocks = [src[groups[src] == run] for run in runs]
        assert all(len(block) == 15 for block in blocks)
        draw = np.random.default_rng([416, seed, index]).permutation(len(blocks))
        fixed[rows] = np.concatenate(blocks)
        moved[rows] = np.concatenate([blocks[i] for i in draw])
        assert not np.array_equal(fixed[rows], moved[rows])
        assert not np.array_equal(moved[rows], src) and not np.array_equal(fixed[rows], src)
        for p in (fixed, moved):
            assert np.array_equal(np.sort(p[rows]), rows)
            assert (np.ptp(groups[p[rows]].reshape(-1, 15), axis=1) == 0).all()
    orders[f'fixed{seed}'], orders[f'moved{seed}'] = fixed, moved
np.savez('orders.npz', **orders)
source = (old/'launch.sbatch').read_text()
source = source.replace('python worker.py', 'python deliver.py').replace('ni26-qual385','ni26-scope416').replace('-t 12','-t 20')
Path('launch.sbatch').write_text(source)
source = (old/'collect.sbatch').read_text().replace('qual385','scope416').replace('collect.py','score.py')
Path('score.sbatch').write_text(source)
pin = ['prepare.py','deliver.py','score.py','launch.sbatch','score.sbatch','orders.npz',
       'data.py','worker.py','replay.py','members.sha256','inputs.sha256',manifest]
Path('sources.sha256').write_text(''.join(f'{hashlib.sha256(Path(p).read_bytes()).hexdigest()}  {p}\n' for p in pin))
