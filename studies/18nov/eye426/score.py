"""Paired external word-task selectivity and independent intertrial qualification."""
import hashlib
import json
from pathlib import Path

import numpy as np

PRIOR = Path('../ni26_select415_20261118')
MODES = ('original', 'aligned', 'reversed')


def interval(x):
    n, o = x.shape
    rng = np.random.default_rng(426)
    p = rng.integers(n, size=(10000, n, 1))
    s = rng.integers(o, size=(10000, n, o))
    return [x.mean(), *np.quantile(x[p, s].mean((1, 2)), [.1, .9])]


def main():
    meta = json.loads((PRIOR/'result.json').read_text())
    digest = hashlib.sha256()
    prior = {}
    for arm in ('old','cap'):
        for i in range(12):
            data = (PRIOR/f'{arm}_{i}.json').read_bytes()
            digest.update(data)
            for r in json.loads(data):
                if r['kind'] == 'W':
                    prior[arm,r['person'],r['seed']] = r
    assert meta['prediction_sha256'] == digest.hexdigest()
    with np.load('../neural-interfaces-2026/data/shin2018/windows.npz') as z:
        y, runs, subjects, first = (z[k] for k in ('y','run','subject','win'))
    labels = np.full((26,2,2,3,60), -1, dtype=int)
    values = np.full(labels.shape[:-1], np.nan)
    quality, power = [], []
    for i in range(26):
        person = i+1
        ids = np.flatnonzero((subjects == f'Shin2018/{person}') & (first == 0))
        with np.load(f'person{person}.npz') as z:
            quality.append(z['quality'])
            power.append(z['rms'].item())
        for a, arm in enumerate(('old','cap')):
            recs = json.loads(Path(f'score-{arm}-{person}.json').read_text())
            assert len(recs) == 6
            for r in recs:
                assert (r['arm'], r['person']) == (arm, person)
                seed = r['seed']
                m = MODES.index(r['mode'])
                index = (i,(415,1415).index(seed),a,m)
                perm = np.random.default_rng([415,person,seed,0]).permutation(60)
                ref = prior[arm,person,seed]
                assert np.array_equal(ids[perm], r['raw']) and ref['rows'] == r['rows'] == perm.tolist()
                assert r['raw'] == ref['raw'] and np.isnan(values[index])
                if m == 0:
                    assert r['labels'] == ref['labels']
                v = np.array(r['labels'])[np.argsort(perm)]
                labels[index] = v
                signed = (v == 2).astype(float) - (v == 1)
                diffs = []
                for run in ('1','2','3'):
                    waves = [signed[(runs[ids] == run) & (y[ids] == c)] for c in (0,1)]
                    assert all(len(w) == 10 for w in waves)
                    diffs.append(waves[1].mean() - waves[0].mean())
                values[index] = 100*np.mean(diffs)
    assert np.isin(labels,[0,1,2]).all() and np.isfinite(values).all()
    rel = values[:,:,1] - values[:,:,0]
    np.testing.assert_allclose(rel[:,:,0].mean(1), meta['W']['per_person'], atol=1e-10)
    quality = np.array(quality)
    output = dict(source_prediction_sha256=digest.hexdigest(), rms_uv=power,
                  intertrial_error_ratio=quality.tolist(), persons=values.tolist(), groups={})
    for group, sl in [('odd',slice(0,None,2)),('even',slice(1,None,2))]:
        d = rel[sl]
        q = quality[sl].mean(1)
        output['groups'][group] = dict(
            archive_scores={arm: {mode: interval(values[sl,:,a,m]) for m,mode in enumerate(MODES)}
                            for a,arm in enumerate(('old','cap'))},
            relative={mode: interval(d[:,:,m]) for m,mode in enumerate(MODES)},
            attenuation=interval(d[:,:,0]-d[:,:,1]), timing=interval(d[:,:,2]-d[:,:,1]),
            intertrial_error_ratio=q.mean(0).tolist(), improved=int((q[:,0] < q[:,1]).sum()),
            changed_fraction=(labels[sl,:,:,1:] != labels[sl,:,:,0:1]).mean(axis=(0,1,4)).tolist())
    Path('result.json').write_text(json.dumps(output,indent=2)+'\n')


if __name__ == '__main__':
    main()
