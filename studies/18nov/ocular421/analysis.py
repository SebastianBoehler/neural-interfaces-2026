"""Pair exact intact EEG package decisions by separately measured eye activity."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression

ROOT = Path('../neural-interfaces-2026')
PRIOR = Path('../ni26_select415_20261118')


def features(directory, run, onset, y):
    stem = directory / f'word{run}'
    header = stem.with_suffix('.vhdr').read_text()
    assert all(s in header for s in ('SamplingInterval=1000\n', 'NumberOfChannels=30\n',
                                     'DataOrientation=MULTIPLEXED', 'BinaryFormat=INT_16'))
    channels = re.findall(r'^Ch(\d+)=([^,]*),[^,]*,([^,]*),([^,\r\n]*)', header, re.M)
    eyes = [(int(i)-1, float(scale)) for i, name, scale, unit in channels
            if name in ('HEOG', 'VEOG') and unit == 'µV']
    assert len(channels) == 30 and [channels[i][1] for i, _ in eyes] == ['HEOG', 'VEOG']
    markers = re.findall(r'^Mk\d+=Stimulus,S (16|32),(\d+),', stem.with_suffix('.vmrk').read_text(), re.M)
    times = np.array([int(t)-1 for _, t in markers])
    assert np.array_equal(times, np.rint(onset * 1000))
    assert np.array_equal(y, [m == '16' for m, _ in markers]) and y.sum() == 10
    signal = np.memmap(stem.with_suffix('.eeg'), dtype='<i2', mode='r').reshape(-1, 30)
    cols, scales = zip(*eyes)
    result = []
    for t in times:
        assert t >= 3000 and t+5000 <= len(signal)
        baseline = np.median(signal[t-3000:t-2000, cols], axis=0) * scales
        bins = (signal[t+1000:t+5000, cols].reshape(200, 20, 2).mean(1) * scales)
        result.append(np.concatenate([np.median(bins, axis=0)-baseline,
                      np.diff(np.percentile(bins, [5, 95], axis=0), axis=0)[0],
                      np.mean(np.diff(bins, axis=0)**2, axis=0)**.5,
                      bins[100:].mean(0)-bins[:100].mean(0)]))
    x = np.asarray(result)
    assert np.isfinite(x).all() and np.all(x.std(0) > 0)
    return (x-x.mean(0))/x.std(0)


def pairs(x, y):
    a, b = (np.flatnonzero(y == c) for c in (1, 0))
    d = cdist(x[a], x[b], 'sqeuclidean')
    cost = np.zeros((15, 15))
    cost[:10, :10] = d
    cost[10:, 10:] = np.inf
    row, col = linear_sum_assignment(cost)
    real = (row < 10) & (col < 10)
    i, j = row[real], col[real]
    assert len(i) == 5
    return np.r_[a[i], b[j]], [d[i, j].mean(), d.mean()]


def archives(d):
    trace = hashlib.sha256()
    bank = {}
    for arm in ('old', 'cap'):
        for shard in range(12):
            path = PRIOR / f'{arm}_{shard}.json'
            raw = path.read_bytes()
            trace.update(raw)
            for r in json.loads(raw):
                if r['kind'] != 'W':
                    continue
                p, s = r['person'], r['seed']
                ids = np.flatnonzero((d['subject'] == f'Shin2018/{p}') & (d['win'] == 0))
                perm = np.random.default_rng([415, p, s, 0]).permutation(60)
                assert np.array_equal(ids[perm], r['raw']) and np.array_equal(perm, r['rows'])
                assert np.array_equal(d['run'][ids[perm]], [g.split('|')[1] for g in r['groups']])
                labels = np.array(r['labels'])[np.argsort(perm)]
                assert np.isin(labels, (0, 1, 2)).all()
                bank[arm,p,s] = (labels == 2).astype(float) - (labels == 1)
    assert trace.hexdigest() == json.loads(Path('reference.json').read_text())['prediction_sha256']
    return bank


def interval(values):
    rng = np.random.default_rng(421)
    n, o = values.shape
    people = rng.integers(n, size=(10000, n, 1))
    orders = rng.integers(o, size=(10000, n, o))
    return [values.mean(), *np.quantile(values[people, orders].mean((1, 2)), [.1, .9])]


def main(smoke):
    with np.load(ROOT/'data/shin2018/windows.npz') as z:
        d = {k: z[k] for k in ('y', 'onset', 'subject', 'run', 'win', 'classes')}
    assert d['classes'].tolist() == ['baseline', 'WG']
    bank = archives(d)
    groups, covariates = [], []
    for person in range(1, 2 if smoke else 27):
        rows = np.flatnonzero((d['subject'] == f'Shin2018/{person}') & (d['win'] == 0))
        group = []
        for run in range(1, 2 if smoke else 4):
            ix = np.flatnonzero(d['run'][rows] == str(run))
            ids, directory = rows[ix], ROOT / f'data/shin2018/raw/VP{person:03d}'
            y = d['y'][ids]
            x = features(directory, run, d['onset'][ids], y)
            selected, dist = pairs(x, y)
            covariates.append((person, x, y, selected, dist))
            pred = np.array([[bank[arm,person,s][ix] for arm in ('old','cap')]
                             for s in (415, 1415)])
            contrasts = []
            for sub in (np.arange(20), selected):
                contrasts.append(100*(pred[:,:,sub[y[sub]==1]].mean(2)-pred[:,:,sub[y[sub]==0]].mean(2)))
            group.append(contrasts)
        groups.append(np.mean(group, axis=0))
    if smoke:
        print('OK EOG marker, real features, archive order and matching')
        return
    data = np.array(groups)  # person x subset x order x archive
    delta = data[:,:,:,1] - data[:,:,:,0]
    ref = json.loads(Path('reference.json').read_text())['W']
    assert np.allclose(delta[:,0].mean(1), ref['per_person'])
    result = {}
    for name, parity in [('primary', 1), ('reserve', 0)]:
        pick = np.arange(1,27)%2 == parity
        q = delta[pick]
        dist = np.mean([r[4] for r in covariates if r[0]%2 == parity],axis=0)
        result[name] = dict(original=interval(q[:,0]), matched=interval(q[:,1]),
                            attenuation=interval(q[:,0]-q[:,1]), absolute=data[pick].mean((0,2)).tolist(),
                            distance_ratio=dist[0]/dist[1], per_person=q.mean(2).tolist())
    train = [r for r in covariates if r[0]%2]
    test = [r for r in covariates if r[0]%2 == 0]
    model = LogisticRegression(C=1, max_iter=500).fit(np.vstack([r[1] for r in train]),np.hstack([r[2] for r in train]))
    scores = []
    for _,x,y,selected,_ in test:
        correct = model.predict(x)==y
        scores.append([correct.mean(),correct[selected].mean()])
    scores = 100*np.array(scores).reshape(13,3,2).mean(1)
    result['reserve_eog_only_accuracy'] = [interval(scores[:,i,None]) for i in range(2)]
    result['intact_delta'] = delta[:,0].mean()
    Path('result.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    main(parser.parse_args().smoke)
