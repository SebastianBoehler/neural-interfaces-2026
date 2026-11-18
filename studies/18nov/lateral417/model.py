"""Broadband hemispheric balance and two-state marginal-likelihood orientation."""
import numpy as np
from scipy.special import expit, logsumexp, softmax


def powers(x):
    x = x.astype(np.float64)
    x -= x.mean(1, keepdims=True)
    x -= x.mean(-1, keepdims=True)
    psd = abs(np.fft.rfft(x * np.hanning(480))) ** 2
    f = np.fft.rfftfreq(480, 1 / 120)
    return np.stack([np.log(np.maximum(psd[..., (f >= lo) & (f < hi)].mean(-1), 1e-12))
                     for lo, hi in ((4, 8), (8, 13), (13, 20), (20, 30))], axis=-1)


def parts(power, names, seed=None):
    left = [i for i, n in enumerate(names) if n[-1].isdigit() and int(n[-1]) % 2]
    right = [names.index(names[i][:-1] + str(int(names[i][-1]) + 1)) for i in left]
    mid = [i for i, n in enumerate(names) if n.endswith('z')]
    assert len(left) == 18 and len(mid) == 5 and len(set(left + right + mid)) == 41
    if seed is not None:
        shuffled = np.random.default_rng(seed).permutation(left + right)
        left, right = np.split(shuffled, 2)
    n = len(power)
    even = np.concatenate(((power[:, left] + power[:, right]) / 2, power[:, mid]), axis=1)
    return even.reshape(n, -1), (power[:, left] - power[:, right]).reshape(n, -1)


def emissions(partition, rows, labels, weight):
    blocks = []
    for i, x in enumerate(partition):
        train = x[rows]
        means = np.array([np.average(train[labels == c], axis=0, weights=weight[labels == c])
                          for c in range(3)])
        var = np.maximum(np.average((train - means[labels])**2, axis=0, weights=weight), 1e-8)
        centers = (means, 2 * means.mean(0) - means) if i else (means, means)
        blocks.append(np.stack([-.5 * (((x[:, None] - m)**2) / var).mean(-1)
                                for m in centers], axis=1))
    return sum(blocks)


def decode(logp):
    margin = logsumexp(logp[:, 0], axis=1) - logsumexp(logp[:, 1], axis=1)
    odds = 0.
    history = []
    for delta in margin:
        history.append(expit(odds))
        odds = np.clip(odds + delta, -5., 5.)
    post = softmax(logp, axis=2)
    def mix(w):
        prior = np.stack((w, 1 - w), axis=1)
        return softmax(logsumexp(logp + np.log(prior[:, :, None]), axis=1), axis=1)
    return dict(native=post[:, 0], reverse=post[:, 1], equal=mix(np.full(len(logp), .5)),
                latent=mix(np.array(history))), np.array(history)


def temperature(partition, rows, labels, weight, people):
    grid = np.array([1, 2, 4, 8, 16, 32, 64])
    losses = []
    assert len(np.unique(people)) == 6
    for p in np.unique(people):
        held = people == p
        logits = emissions(partition, rows[~held], labels[~held], weight[~held])[rows[held], 0]
        logits = logits[None] * grid[:, None, None]
        nll = logsumexp(logits, axis=2) - logits[:, np.arange(held.sum()), labels[held]]
        losses.append(np.average(nll, axis=1, weights=weight[held]))
    loss = np.mean(losses, axis=0)
    return int(grid[loss.argmin()]), loss.tolist()
