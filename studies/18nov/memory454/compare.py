"""Paired effects of training on coherent spectral gains on committed §454 decisions."""
import json
from pathlib import Path
import sys

import numpy as np

from score import official, previous, ref, sha


def compare(seed):
    original = ref.WORK/'ni26_memory454_20261118'
    branches = {'coherent':Path.cwd(), 'scattered':Path.cwd(), 'kernel':original}
    banks, hashes = {}, {}
    for kind, root in branches.items():
        path = root / f'{kind}_{seed}_decisions.npz'
        result = json.loads((root / f'{kind}_{seed}_result.json').read_text())
        assert result['decisions'] == sha(path)
        hashes[kind] = result['decisions']
        with np.load(path) as z:
            banks[kind] = {k:z[k] for k in z.files}
    for k in banks['coherent']:
        if k.endswith('|source'):
            assert all(np.array_equal(b[k],banks['coherent'][k]) for b in banks.values())
    with np.load(ref.BASE/'data/proteus_windows.npz') as data:
        labels = data['y']
        meta = {k:data[k] for k in ('subject','session','interface')}
    meta['person'] = np.array([p.split('/')[-1] for p in meta['subject']])
    meta['stream'] = np.char.add(np.char.add(meta['subject'],'|'),meta['session'])
    metric = official()
    values = {a:np.empty((10,8)) for a in banks}
    counts = []
    for pi,p in enumerate(ref.PEOPLE):
        mask = meta['person'] == p
        counts.append(len(set(zip(meta['stream'][mask],meta['interface'][mask]))))
        for oi,order in enumerate(ref.ORDERS):
            for name,bank in banks.items():
                result = metric(labels[mask],bank[f'{p}|{order}|memory'],meta,mask)
                assert result['n_cells'] == counts[-1]
                values[name][pi,oi] = 100*result['balanced_accuracy_cell_mean']
    assert sum(counts) == 28
    contrasts = {name:previous.interval(values['coherent']-values[name],np.array(counts))
                 for name in ('kernel','scattered')}
    Path(f'gain_{seed}_comparison.json').write_text(json.dumps(dict(contrasts=contrasts,hashes=hashes), indent=2))


if __name__ == '__main__':
    compare(int(sys.argv[1]))
