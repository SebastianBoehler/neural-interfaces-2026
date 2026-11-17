"""Source-only polarity-invariant spatial states and within-window dynamics (§328)."""
import numpy as np

K = 16


def topographies(raw):
    x = np.asarray(raw, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (41, 480) or not np.isfinite(x).all():
        raise ValueError('finite original N×41×480 tensor required')
    x = x - x.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(x, axis=-1)
    freq = np.fft.rfftfreq(480, 1 / 120)
    spectrum[..., (freq < 2) | (freq > 30)] = 0
    x = np.fft.irfft(spectrum, n=480, axis=-1).transpose(0, 2, 1)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, 1e-12), norm[..., 0]


def unit(x):
    x = x - x.mean(axis=-1, keepdims=True)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(norm < 1e-12): raise ValueError('empty microstate direction')
    return x / norm


def fit(raw, people):
    x, gfp = topographies(raw)
    rng = np.random.default_rng(328)
    peaks = []
    for person in sorted(set(people)):
        a, g = x[people == person], gfp[people == person]
        mask = (g[:, 1:-1] > g[:, :-2]) & (g[:, 1:-1] >= g[:, 2:])
        values = a[:, 1:-1][mask]
        if len(values) < 2048: raise ValueError('fewer than 2048 source peaks/person')
        peaks.append(values[rng.choice(len(values), 2048, replace=False)])
    values = np.concatenate(peaks)
    centers = unit(values[rng.choice(len(values), K, replace=False)].copy())
    for _ in range(30):
        corr = values @ centers.T
        assignment = np.abs(corr).argmax(axis=1)
        updated = []
        for state in range(K):
            mask = assignment == state
            if not mask.any(): raise ValueError('empty microstate cluster')
            signs = np.where(corr[mask, state] >= 0, 1., -1.)
            updated.append(np.mean(values[mask] * signs[:, None], axis=0))
        centers = unit(np.stack(updated))
    random = unit(np.random.default_rng(328).normal(size=(K, 41)))
    return centers, random


def tokenize(raw, centers):
    x, _ = topographies(raw)
    return np.abs(x @ centers.T).argmax(axis=-1)


def summarize(tokens):
    if tokens.ndim != 2 or tokens.shape[1] != 480: raise ValueError('480 tokens/window required')
    result = []
    for row in tokens:
        occupancy = np.bincount(row, minlength=K) / 480
        transitions = np.bincount(K * row[:-1] + row[1:], minlength=K*K) / 479
        starts = np.r_[0, np.flatnonzero(row[1:] != row[:-1]) + 1]
        durations = np.diff(np.r_[starts, 480])
        counts = np.bincount(row[starts], minlength=K)
        dwell = np.bincount(row[starts], weights=durations, minlength=K) / np.maximum(counts, 1)
        result.append(np.r_[occupancy, transitions, dwell])
    return np.asarray(result)


def raw287(raw, scale):
    x = raw.astype(np.float64) / scale
    p = np.abs(np.fft.rfft(x, axis=-1))**2 / 480**2
    f = np.fft.rfftfreq(480, 1 / 120)
    bands = ((.1,4),(4,8),(8,13),(13,20),(20,30),(30,55))
    return np.concatenate([x.mean(axis=-1)] + [np.log(np.maximum(p[..., (f>=a)&(f<b)].mean(axis=-1), 1e-12)) for a,b in bands], axis=1)


def features(raw, centers, random, scale):
    learned = tokenize(raw, centers)
    shuffled = learned[:, np.random.default_rng(328).permutation(480)]
    ls, ss, rs = summarize(learned), summarize(shuffled), summarize(tokenize(raw, random))
    if not np.array_equal(ls[:, :K], ss[:, :K]): raise ValueError('shuffle changed occupancy')
    rawblock = raw287(raw, scale)
    return dict(raw287=rawblock, occupancy=ls[:, :K], raw_occupancy=np.c_[rawblock, ls[:, :K]],
        ordered=np.c_[rawblock, ls], shuffled=np.c_[rawblock, ss], random=np.c_[rawblock, rs])
