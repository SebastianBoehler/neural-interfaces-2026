"""Independent P01 NumPy/SciPy validation of both saved CUDA kernel mixtures."""
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist, pdist, squareform
from score379 import BASE, DATA_SHA, ORDERS, sha
from source_labels import read_training_labels


def main():
    file = BASE/'data/proteus_windows.npz'
    assert sha(file) == DATA_SHA
    with np.load(file) as d:
        person = np.array([s.split('/')[-1] for s in d['subject']])
        fit, test = np.flatnonzero(person != 'P01'), np.flatnonzero(person == 'P01')
        cells = np.char.add(d['session'][fit], d['interface'][fit])
        raw = d['X'].astype(np.float64)
    raw -= raw.mean(1, keepdims=True)
    raw -= raw.mean(2, keepdims=True)
    freq = np.fft.rfftfreq(480, 1/120)
    spec = np.fft.rfft(raw * np.hanning(480), axis=-1)
    del raw
    tri = np.triu_indices(41, 1)
    train, query = [], []
    for lo, hi in [(4,8),(8,12),(12,18),(18,25),(25,36)]:
        z = spec[:, :, (freq >= lo) & (freq < hi)]
        cov = np.einsum('nif,njf->nij', z, z.conj()).real
        power = cov.diagonal(axis1=1, axis2=2).clip(1e-15)
        corr = cov[:, tri[0], tri[1]] / np.sqrt(power[:,tri[0]]*power[:,tri[1]])
        for v in [np.log(power), corr]:
            d = pdist(v[fit], 'sqeuclidean')
            middle = (len(d)-1)//2
            scale = np.partition(d,middle)[middle]
            train.append(np.exp(-squareform(d)/scale))
            query.append(np.exp(-cdist(v[test],v[fit],'sqeuclidean')/scale))
    y = read_training_labels(file, fit, len(person))
    key = [f'{p}|{cell}|{label}' for p,cell,label in zip(person[fit],cells,y)]
    _, inv, count = np.unique(key, return_inverse=True, return_counts=True)
    w = 1/count[inv]
    differences = {}
    for seed in (405,1405):
        root = Path(str(seed))
        info = json.loads((root/'P01.json').read_text())
        assert sha(root/'P01.npz') == info['outputs_sha256']
        mix = np.array(info['weights'])
        with np.load(root/'P01.npz') as archive:
            assert np.array_equal(archive['index'],test)
            for arm, weights in [('U', np.full(10,.1)), ('L', mix), ('D',np.roll(mix.reshape(5,2),2,axis=0).ravel())]:
                k = sum(v * a for v,a in zip(train,weights))
                k.flat[::len(k)+1] += w.mean()/w
                alpha = solve(k, np.eye(3)[y], assume_a='pos')
                pred = (sum(v * a for v,a in zip(query,weights)) @ alpha).argmax(1)
                differences[f'{seed}/{arm}'] = sum(int((pred!=archive[f'{arm}|{o}']).sum()) for o in ORDERS)
    assert not any(differences.values()), differences
    Path('audit405.json').write_text(json.dumps(dict(rows=len(test), differences=differences),indent=2))


if __name__=='__main__':
    main()
